//! e-means clustering: mini-batch EMA centroids.
//!
//! This follows Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze and Jégou,
//! *e-means: outperforming k-means (in one-tenth the epochs)*, arXiv:2611.09501,
//! and the reference implementation in Faiss `contrib/e_means`.

pub(crate) mod assignment;
mod laws;
mod update;
mod util;

use assignment::{
    nearest_4x_f32_3_dot, nearest_4x_f32_4_dot, nearest_4x_f32_generic, nearest_4x_f64_generic,
    nearest_centroid_f32_generic, nearest_f32_3_dot, nearest_f32_4_dot, nearest_f64_panel,
};
use laws::{resolve_alpha0, resolve_batch_size, scheduled_alpha};
use update::{
    apply_batch, pack_centroids_f32_dot_panels, pack_centroids_f32_panels,
    pack_centroids_f64_panels, revive_unused,
};
use util::{Scramble, shuffle};

use fearless_simd::{Level, Simd, dispatch};

use crate::kmeans_triangle::{Random, get_centroids_with_random};

/// Passes over the data when the caller does not choose.
pub const DEFAULT_EPOCHS: usize = 5;

/// Revive under-used clusters, measured in batch updates.
const CHECK_REVIVAL_EVERY: usize = 5;

/// Initial mass.
const INIT_USAGE: f64 = 1e-10;

/// A cluster below this fraction of the mean mass is revived.
const REVIVAL_THRESHOLD: f64 = 0.1;

/// Revival hurt held-out error at two dimensions in the reference implementation.
const NO_REVIVAL_MAX_DIMENSIONS: usize = 2;

/// Split a heavy row so no chunk is larger than this fraction of a batch.
const CHUNKS_PER_BATCH: f64 = 16.0;

/// Maximum exact `f32` chunk weight for zero-gather inline chunks.
const MAX_INLINE_CHUNK_WEIGHT: f64 = 1_000_000.0;

/// Optional inputs to the core.
#[derive(Clone, Copy, Default)]
pub struct EmeansOptions<'a> {
    /// A weight per row, for input where many points are exactly equal.
    pub weights: Option<&'a [f64]>,

    /// Replaces the random draw, for comparing runs from one start.
    pub initial_centroids: Option<&'a [f64]>,

    /// Overrides the scaling law. Clamped to the number of points.
    pub batch_size: Option<usize>,

    /// Overrides the dimension-dependent revival threshold. Zero disables revival.
    pub revival_threshold: Option<f64>,

    /// Records the last-epoch assignment of every row.
    pub capture_assignments: bool,

    /// Explicit deterministic seed for initialization and epoch shuffling.
    pub seed: Option<u32>,

    /// Uses 4-lane `f32x4` centroid assignment while keeping EMA state in `f64`.
    pub use_f32_assignment: bool,

    /// Uses `0.5 * ||c||^2 - <p, c>` algebraic scoring and zero-gather inline
    /// chunks for bounded packed inputs (`0..=255`).
    pub use_dot_product: bool,

    /// Optional pre-converted `f32` points buffer to avoid widening-then-narrowing copies.
    pub points_f32: Option<&'a [f32]>,
}

/// The outcome of one e-means run.
pub struct EmeansResult {
    pub centroids: Vec<f64>,
    pub dimensions: usize,
    pub epochs: usize,
    pub updates: usize,
    pub batch_size: usize,
    pub alpha0: f64,
    pub assignments: Option<Vec<u32>>,
}

impl EmeansResult {
    pub fn centroid_rows(&self) -> Vec<Vec<f64>> {
        self.centroids
            .chunks_exact(self.dimensions)
            .map(<[f64]>::to_vec)
            .collect()
    }

    pub fn assignments(&self) -> &[u32] {
        self.assignments.as_deref().unwrap_or(&[])
    }
}

/// Runs e-means once at the strongest SIMD level this build supports.
pub fn emeans_dispatched(
    k: usize,
    epochs: usize,
    points: &[f64],
    dimensions: usize,
    options: EmeansOptions<'_>,
) -> EmeansResult {
    let level = Level::baseline();
    dispatch!(level, simd => emeans(
        simd,
        k,
        epochs,
        points,
        dimensions,
        options
    ))
}

/// e-means over a flat row-major buffer.
pub fn emeans<S: Simd>(
    simd: S,
    k: usize,
    epochs: usize,
    points: &[f64],
    dimensions: usize,
    options: EmeansOptions<'_>,
) -> EmeansResult {
    let point_count = points.len().checked_div(dimensions).unwrap_or(0);
    if point_count == 0 || dimensions == 0 || k == 0 || epochs == 0 {
        return EmeansResult {
            centroids: Vec::new(),
            dimensions,
            epochs: 0,
            updates: 0,
            batch_size: 0,
            alpha0: 0.0,
            assignments: options.capture_assignments.then(Vec::new),
        };
    }

    let weights = options.weights;
    debug_assert!(weights.is_none_or(|all| all.len() >= point_count));

    let total_weight: f64 = match weights {
        Some(all) => all.iter().take(point_count).sum(),
        None => point_count as f64,
    };
    let total_weight = total_weight.max(f64::MIN_POSITIVE);

    let batch_size = resolve_batch_size(total_weight, k, epochs, dimensions, options.batch_size);
    let alpha0 = resolve_alpha0(total_weight, k, epochs, batch_size);
    let revival_threshold = options.revival_threshold.unwrap_or(
        if dimensions <= NO_REVIVAL_MAX_DIMENSIONS || k > point_count {
            0.0
        } else {
            REVIVAL_THRESHOLD
        },
    );

    let mut entropy = match options.seed {
        Some(seed) => Random::from_seed(seed),
        None => Random::new(),
    };
    let mut centroids = match options.initial_centroids {
        Some(given) => {
            debug_assert_eq!(given.len(), k * dimensions);
            given.to_vec()
        }
        None => {
            get_centroids_with_random(points, dimensions, point_count, k, weights, &mut entropy)
        }
    };
    let mut random = Scramble::from_random(&mut entropy);
    let mut usage = vec![INIT_USAGE; k];

    let batches_per_epoch = (total_weight / batch_size as f64).ceil().max(1.0) as usize;
    let total_steps = epochs.saturating_mul(batches_per_epoch);
    let mut updates = 0usize;

    let batch_capacity = batch_size as f64;
    let chunk_capacity = (batch_capacity / CHUNKS_PER_BATCH).max(1.0);
    let mut assignments = options.capture_assignments.then(|| vec![0u32; point_count]);

    // Counts and sums for the batch and final pass share one allocation.
    let mut stats = vec![0.0f64; 2 * k + 2 * k * dimensions];
    let (batch_counts, rest) = stats.split_at_mut(k);
    let (batch_sums, rest) = rest.split_at_mut(k * dimensions);
    let (pass_counts, pass_sums) = rest.split_at_mut(k);
    let mut touched = Vec::with_capacity(k.min(point_count));

    let inline_eligible = options.use_f32_assignment
        && options.use_dot_product
        && !options.capture_assignments
        && chunk_capacity <= MAX_INLINE_CHUNK_WEIGHT
        && matches!(dimensions, 3 | 4);

    if inline_eligible && dimensions == 3 {
        let mut chunks = build_inline_chunks_3(points, weights, point_count, chunk_capacity);
        if chunks.is_empty() {
            return EmeansResult {
                centroids,
                dimensions,
                epochs,
                updates,
                batch_size,
                alpha0,
                assignments,
            };
        }

        let k_panels = k.div_ceil(4);
        let mut panels = vec![f32::INFINITY; k_panels * 16];
        pack_centroids_f32_dot_panels(&centroids, 3, &mut panels);

        for epoch in 0..epochs {
            shuffle(&mut chunks, &mut random);
            let last_epoch = epoch + 1 == epochs;
            let mut cursor = 0usize;
            let mut remaining = f64::from(chunks[0][3]);

            while cursor < chunks.len() {
                fill_batch_inline_3(
                    simd,
                    &panels,
                    k_panels,
                    &chunks,
                    batch_capacity,
                    &mut cursor,
                    &mut remaining,
                    batch_counts,
                    batch_sums,
                    &mut touched,
                );

                let alpha = scheduled_alpha(alpha0, updates, total_steps);
                apply_batch(
                    &mut centroids,
                    &mut usage,
                    3,
                    batch_counts,
                    batch_sums,
                    &touched,
                    alpha,
                );
                updates += 1;

                accumulate_and_clear_batch(
                    last_epoch,
                    3,
                    &mut touched,
                    batch_counts,
                    batch_sums,
                    pass_counts,
                    pass_sums,
                );

                if updates.is_multiple_of(CHECK_REVIVAL_EVERY) {
                    let _ = revive_unused(
                        &mut centroids,
                        &mut usage,
                        3,
                        revival_threshold,
                        &mut random,
                    );
                }
                pack_centroids_f32_dot_panels(&centroids, 3, &mut panels);
            }
        }
    } else if inline_eligible && dimensions == 4 {
        let mut chunks = build_inline_chunks_4(points, weights, point_count, chunk_capacity);
        if chunks.is_empty() {
            return EmeansResult {
                centroids,
                dimensions,
                epochs,
                updates,
                batch_size,
                alpha0,
                assignments,
            };
        }

        let k_panels = k.div_ceil(4);
        let mut panels = vec![f32::INFINITY; k_panels * 20];
        pack_centroids_f32_dot_panels(&centroids, 4, &mut panels);

        for epoch in 0..epochs {
            shuffle(&mut chunks, &mut random);
            let last_epoch = epoch + 1 == epochs;
            let mut cursor = 0usize;
            let mut remaining = f64::from(chunks[0][4]);

            while cursor < chunks.len() {
                fill_batch_inline_4(
                    simd,
                    &panels,
                    k_panels,
                    &chunks,
                    batch_capacity,
                    &mut cursor,
                    &mut remaining,
                    batch_counts,
                    batch_sums,
                    &mut touched,
                );

                let alpha = scheduled_alpha(alpha0, updates, total_steps);
                apply_batch(
                    &mut centroids,
                    &mut usage,
                    4,
                    batch_counts,
                    batch_sums,
                    &touched,
                    alpha,
                );
                updates += 1;

                accumulate_and_clear_batch(
                    last_epoch,
                    4,
                    &mut touched,
                    batch_counts,
                    batch_sums,
                    pass_counts,
                    pass_sums,
                );

                if updates.is_multiple_of(CHECK_REVIVAL_EVERY) {
                    let _ = revive_unused(
                        &mut centroids,
                        &mut usage,
                        4,
                        revival_threshold,
                        &mut random,
                    );
                }
                pack_centroids_f32_dot_panels(&centroids, 4, &mut panels);
            }
        }
    } else {
        let mut chunks = build_chunks(weights, point_count, chunk_capacity);
        if chunks.is_empty() {
            return EmeansResult {
                centroids,
                dimensions,
                epochs,
                updates,
                batch_size,
                alpha0,
                assignments,
            };
        }

        if options.use_f32_assignment {
            let owned_f32: Vec<f32>;
            let points_f32: &[f32] = match options.points_f32 {
                Some(given) if given.len() >= point_count * dimensions => given,
                _ => {
                    owned_f32 = points.iter().map(|value| *value as f32).collect();
                    &owned_f32
                }
            };
            let k_panels = k.div_ceil(4);
            let mut panels = vec![f32::INFINITY; k_panels * dimensions * 4];
            pack_centroids_f32_panels(&centroids, dimensions, &mut panels);

            for epoch in 0..epochs {
                shuffle(&mut chunks, &mut random);
                let last_epoch = epoch + 1 == epochs;
                let mut cursor = 0usize;
                let mut remaining = chunks[0].1;

                while cursor < chunks.len() {
                    fill_batch_f32(
                        simd,
                        points_f32,
                        &panels,
                        dimensions,
                        k_panels,
                        &chunks,
                        batch_capacity,
                        &mut cursor,
                        &mut remaining,
                        batch_counts,
                        batch_sums,
                        &mut touched,
                        if last_epoch {
                            assignments.as_deref_mut()
                        } else {
                            None
                        },
                    );

                    let alpha = scheduled_alpha(alpha0, updates, total_steps);
                    apply_batch(
                        &mut centroids,
                        &mut usage,
                        dimensions,
                        batch_counts,
                        batch_sums,
                        &touched,
                        alpha,
                    );
                    updates += 1;

                    accumulate_and_clear_batch(
                        last_epoch,
                        dimensions,
                        &mut touched,
                        batch_counts,
                        batch_sums,
                        pass_counts,
                        pass_sums,
                    );

                    if updates.is_multiple_of(CHECK_REVIVAL_EVERY) {
                        let _ = revive_unused(
                            &mut centroids,
                            &mut usage,
                            dimensions,
                            revival_threshold,
                            &mut random,
                        );
                    }
                    pack_centroids_f32_panels(&centroids, dimensions, &mut panels);
                }
            }
        } else {
            let k_pairs = k.div_ceil(2);
            let mut panels = vec![f64::INFINITY; k_pairs * dimensions * 2];
            pack_centroids_f64_panels(&centroids, dimensions, k, &mut panels);

            for epoch in 0..epochs {
                shuffle(&mut chunks, &mut random);
                let last_epoch = epoch + 1 == epochs;
                let mut cursor = 0usize;
                let mut remaining = chunks[0].1;

                while cursor < chunks.len() {
                    fill_batch_f64(
                        simd,
                        points,
                        &panels,
                        dimensions,
                        k_pairs,
                        &chunks,
                        batch_capacity,
                        &mut cursor,
                        &mut remaining,
                        batch_counts,
                        batch_sums,
                        &mut touched,
                        if last_epoch {
                            assignments.as_deref_mut()
                        } else {
                            None
                        },
                    );

                    let alpha = scheduled_alpha(alpha0, updates, total_steps);
                    apply_batch(
                        &mut centroids,
                        &mut usage,
                        dimensions,
                        batch_counts,
                        batch_sums,
                        &touched,
                        alpha,
                    );
                    updates += 1;

                    accumulate_and_clear_batch(
                        last_epoch,
                        dimensions,
                        &mut touched,
                        batch_counts,
                        batch_sums,
                        pass_counts,
                        pass_sums,
                    );

                    if updates.is_multiple_of(CHECK_REVIVAL_EVERY) {
                        let _ = revive_unused(
                            &mut centroids,
                            &mut usage,
                            dimensions,
                            revival_threshold,
                            &mut random,
                        );
                    }
                    pack_centroids_f64_panels(&centroids, dimensions, k, &mut panels);
                }
            }
        }
    }

    finalize_centroids(pass_counts, pass_sums, &mut centroids, dimensions);

    EmeansResult {
        centroids,
        dimensions,
        epochs,
        updates,
        batch_size,
        alpha0,
        assignments,
    }
}

fn accumulate_and_clear_batch(
    last_epoch: bool,
    dimensions: usize,
    touched: &mut Vec<usize>,
    batch_counts: &mut [f64],
    batch_sums: &mut [f64],
    pass_counts: &mut [f64],
    pass_sums: &mut [f64],
) {
    for &centroid in touched.iter() {
        let base = centroid * dimensions;
        if last_epoch {
            pass_counts[centroid] += batch_counts[centroid];
            for part in 0..dimensions {
                pass_sums[base + part] += batch_sums[base + part];
            }
        }
        batch_counts[centroid] = 0.0;
        batch_sums[base..base + dimensions].fill(0.0);
    }
    touched.clear();
}

#[allow(clippy::too_many_arguments)]
fn fill_batch_inline_3<S: Simd>(
    simd: S,
    panels: &[f32],
    k_panels: usize,
    chunks: &[[f32; 4]],
    batch_capacity: f64,
    cursor: &mut usize,
    remaining: &mut f64,
    batch_counts: &mut [f64],
    batch_sums: &mut [f64],
    touched: &mut Vec<usize>,
) {
    let mut batch_weight = 0.0f64;

    while *cursor + 4 <= chunks.len() {
        let c0 = chunks[*cursor];
        let w0 = f64::from(c0[3]);
        if *remaining != w0 {
            break;
        }
        let c1 = chunks[*cursor + 1];
        let c2 = chunks[*cursor + 2];
        let c3 = chunks[*cursor + 3];
        let w1 = f64::from(c1[3]);
        let w2 = f64::from(c2[3]);
        let w3 = f64::from(c3[3]);
        let sum_w = w0 + w1 + w2 + w3;
        if batch_weight + sum_w > batch_capacity {
            break;
        }

        let assigned = nearest_4x_f32_3_dot(
            simd,
            [
                [c0[0], c0[1], c0[2]],
                [c1[0], c1[1], c1[2]],
                [c2[0], c2[1], c2[2]],
                [c3[0], c3[1], c3[2]],
            ],
            panels,
            k_panels,
        );

        let items = [
            (c0, w0, assigned[0]),
            (c1, w1, assigned[1]),
            (c2, w2, assigned[2]),
            (c3, w3, assigned[3]),
        ];
        for (chunk, take, centroid) in items {
            if batch_counts[centroid] == 0.0 {
                touched.push(centroid);
            }
            batch_counts[centroid] += take;
            let sum_offset = centroid * 3;
            batch_sums[sum_offset] += take * f64::from(chunk[0]);
            batch_sums[sum_offset + 1] += take * f64::from(chunk[1]);
            batch_sums[sum_offset + 2] += take * f64::from(chunk[2]);
        }

        batch_weight += sum_w;
        *cursor += 4;
        *remaining = if *cursor < chunks.len() {
            f64::from(chunks[*cursor][3])
        } else {
            0.0
        };
    }

    while *cursor < chunks.len() && batch_weight < batch_capacity {
        let chunk = chunks[*cursor];
        let take = (*remaining).min(batch_capacity - batch_weight);
        let centroid = nearest_f32_3_dot(simd, &chunk[..3], panels, k_panels);

        if batch_counts[centroid] == 0.0 {
            touched.push(centroid);
        }
        batch_counts[centroid] += take;
        let sum_offset = centroid * 3;
        batch_sums[sum_offset] += take * f64::from(chunk[0]);
        batch_sums[sum_offset + 1] += take * f64::from(chunk[1]);
        batch_sums[sum_offset + 2] += take * f64::from(chunk[2]);

        batch_weight += take;
        *remaining -= take;
        if *remaining <= 0.0 {
            *cursor += 1;
            if *cursor < chunks.len() {
                *remaining = f64::from(chunks[*cursor][3]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_batch_inline_4<S: Simd>(
    simd: S,
    panels: &[f32],
    k_panels: usize,
    chunks: &[[f32; 5]],
    batch_capacity: f64,
    cursor: &mut usize,
    remaining: &mut f64,
    batch_counts: &mut [f64],
    batch_sums: &mut [f64],
    touched: &mut Vec<usize>,
) {
    let mut batch_weight = 0.0f64;

    while *cursor + 4 <= chunks.len() {
        let c0 = chunks[*cursor];
        let w0 = f64::from(c0[4]);
        if *remaining != w0 {
            break;
        }
        let c1 = chunks[*cursor + 1];
        let c2 = chunks[*cursor + 2];
        let c3 = chunks[*cursor + 3];
        let w1 = f64::from(c1[4]);
        let w2 = f64::from(c2[4]);
        let w3 = f64::from(c3[4]);
        let sum_w = w0 + w1 + w2 + w3;
        if batch_weight + sum_w > batch_capacity {
            break;
        }

        let assigned = nearest_4x_f32_4_dot(
            simd,
            [
                [c0[0], c0[1], c0[2], c0[3]],
                [c1[0], c1[1], c1[2], c1[3]],
                [c2[0], c2[1], c2[2], c2[3]],
                [c3[0], c3[1], c3[2], c3[3]],
            ],
            panels,
            k_panels,
        );

        let items = [
            (c0, w0, assigned[0]),
            (c1, w1, assigned[1]),
            (c2, w2, assigned[2]),
            (c3, w3, assigned[3]),
        ];
        for (chunk, take, centroid) in items {
            if batch_counts[centroid] == 0.0 {
                touched.push(centroid);
            }
            batch_counts[centroid] += take;
            let sum_offset = centroid * 4;
            batch_sums[sum_offset] += take * f64::from(chunk[0]);
            batch_sums[sum_offset + 1] += take * f64::from(chunk[1]);
            batch_sums[sum_offset + 2] += take * f64::from(chunk[2]);
            batch_sums[sum_offset + 3] += take * f64::from(chunk[3]);
        }

        batch_weight += sum_w;
        *cursor += 4;
        *remaining = if *cursor < chunks.len() {
            f64::from(chunks[*cursor][4])
        } else {
            0.0
        };
    }

    while *cursor < chunks.len() && batch_weight < batch_capacity {
        let chunk = chunks[*cursor];
        let take = (*remaining).min(batch_capacity - batch_weight);
        let centroid = nearest_f32_4_dot(simd, &chunk[..4], panels, k_panels);

        if batch_counts[centroid] == 0.0 {
            touched.push(centroid);
        }
        batch_counts[centroid] += take;
        let sum_offset = centroid * 4;
        batch_sums[sum_offset] += take * f64::from(chunk[0]);
        batch_sums[sum_offset + 1] += take * f64::from(chunk[1]);
        batch_sums[sum_offset + 2] += take * f64::from(chunk[2]);
        batch_sums[sum_offset + 3] += take * f64::from(chunk[3]);

        batch_weight += take;
        *remaining -= take;
        if *remaining <= 0.0 {
            *cursor += 1;
            if *cursor < chunks.len() {
                *remaining = f64::from(chunks[*cursor][4]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_batch_f32<S: Simd>(
    simd: S,
    points: &[f32],
    panels: &[f32],
    dimensions: usize,
    k_panels: usize,
    chunks: &[(u32, f64)],
    batch_capacity: f64,
    cursor: &mut usize,
    remaining: &mut f64,
    batch_counts: &mut [f64],
    batch_sums: &mut [f64],
    touched: &mut Vec<usize>,
    mut assignments: Option<&mut [u32]>,
) {
    let mut batch_weight = 0.0f64;

    while *cursor + 4 <= chunks.len() {
        let (r0, w0) = chunks[*cursor];
        if *remaining != w0 {
            break;
        }
        let (r1, w1) = chunks[*cursor + 1];
        let (r2, w2) = chunks[*cursor + 2];
        let (r3, w3) = chunks[*cursor + 3];
        let sum_w = w0 + w1 + w2 + w3;
        if batch_weight + sum_w > batch_capacity {
            break;
        }

        let o0 = r0 as usize * dimensions;
        let o1 = r1 as usize * dimensions;
        let o2 = r2 as usize * dimensions;
        let o3 = r3 as usize * dimensions;
        let assigned = nearest_4x_f32_generic(
            simd,
            [
                &points[o0..o0 + dimensions],
                &points[o1..o1 + dimensions],
                &points[o2..o2 + dimensions],
                &points[o3..o3 + dimensions],
            ],
            panels,
            dimensions,
            k_panels,
        );

        let items = [
            (r0 as usize, o0, w0, assigned[0]),
            (r1 as usize, o1, w1, assigned[1]),
            (r2 as usize, o2, w2, assigned[2]),
            (r3 as usize, o3, w3, assigned[3]),
        ];
        for (row, point_offset, take, centroid) in items {
            if batch_counts[centroid] == 0.0 {
                touched.push(centroid);
            }
            batch_counts[centroid] += take;
            let sum_offset = centroid * dimensions;
            for part in 0..dimensions {
                batch_sums[sum_offset + part] += take * f64::from(points[point_offset + part]);
            }
            if let Some(ref mut labels) = assignments {
                labels[row] = centroid as u32;
            }
        }

        batch_weight += sum_w;
        *cursor += 4;
        *remaining = if *cursor < chunks.len() {
            chunks[*cursor].1
        } else {
            0.0
        };
    }

    while *cursor < chunks.len() && batch_weight < batch_capacity {
        let row = chunks[*cursor].0 as usize;
        let take = (*remaining).min(batch_capacity - batch_weight);
        let point_offset = row * dimensions;
        let point = &points[point_offset..point_offset + dimensions];
        let centroid = nearest_centroid_f32_generic(simd, point, panels, dimensions, k_panels);

        if batch_counts[centroid] == 0.0 {
            touched.push(centroid);
        }
        batch_counts[centroid] += take;
        let sum_offset = centroid * dimensions;
        for part in 0..dimensions {
            batch_sums[sum_offset + part] += take * f64::from(point[part]);
        }
        if let Some(ref mut labels) = assignments {
            labels[row] = centroid as u32;
        }

        batch_weight += take;
        *remaining -= take;
        if *remaining <= 0.0 {
            *cursor += 1;
            if *cursor < chunks.len() {
                *remaining = chunks[*cursor].1;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_batch_f64<S: Simd>(
    simd: S,
    points: &[f64],
    panels: &[f64],
    dimensions: usize,
    k_pairs: usize,
    chunks: &[(u32, f64)],
    batch_capacity: f64,
    cursor: &mut usize,
    remaining: &mut f64,
    batch_counts: &mut [f64],
    batch_sums: &mut [f64],
    touched: &mut Vec<usize>,
    mut assignments: Option<&mut [u32]>,
) {
    let mut batch_weight = 0.0f64;

    while *cursor + 4 <= chunks.len() {
        let (r0, w0) = chunks[*cursor];
        if *remaining != w0 {
            break;
        }
        let (r1, w1) = chunks[*cursor + 1];
        let (r2, w2) = chunks[*cursor + 2];
        let (r3, w3) = chunks[*cursor + 3];
        let sum_w = w0 + w1 + w2 + w3;
        if batch_weight + sum_w > batch_capacity {
            break;
        }

        let o0 = r0 as usize * dimensions;
        let o1 = r1 as usize * dimensions;
        let o2 = r2 as usize * dimensions;
        let o3 = r3 as usize * dimensions;
        let assigned = nearest_4x_f64_generic(
            simd,
            [
                &points[o0..o0 + dimensions],
                &points[o1..o1 + dimensions],
                &points[o2..o2 + dimensions],
                &points[o3..o3 + dimensions],
            ],
            panels,
            dimensions,
            k_pairs,
        );

        let items = [
            (r0 as usize, o0, w0, assigned[0]),
            (r1 as usize, o1, w1, assigned[1]),
            (r2 as usize, o2, w2, assigned[2]),
            (r3 as usize, o3, w3, assigned[3]),
        ];
        for (row, point_offset, take, centroid) in items {
            if batch_counts[centroid] == 0.0 {
                touched.push(centroid);
            }
            batch_counts[centroid] += take;
            let sum_offset = centroid * dimensions;
            for part in 0..dimensions {
                batch_sums[sum_offset + part] += take * points[point_offset + part];
            }
            if let Some(ref mut labels) = assignments {
                labels[row] = centroid as u32;
            }
        }

        batch_weight += sum_w;
        *cursor += 4;
        *remaining = if *cursor < chunks.len() {
            chunks[*cursor].1
        } else {
            0.0
        };
    }

    while *cursor < chunks.len() && batch_weight < batch_capacity {
        let row = chunks[*cursor].0 as usize;
        let take = (*remaining).min(batch_capacity - batch_weight);
        let point_offset = row * dimensions;
        let point = &points[point_offset..point_offset + dimensions];
        let centroid = nearest_f64_panel(simd, point, panels, dimensions, k_pairs);

        if batch_counts[centroid] == 0.0 {
            touched.push(centroid);
        }
        batch_counts[centroid] += take;
        let sum_offset = centroid * dimensions;
        for part in 0..dimensions {
            batch_sums[sum_offset + part] += take * point[part];
        }
        if let Some(ref mut labels) = assignments {
            labels[row] = centroid as u32;
        }

        batch_weight += take;
        *remaining -= take;
        if *remaining <= 0.0 {
            *cursor += 1;
            if *cursor < chunks.len() {
                *remaining = chunks[*cursor].1;
            }
        }
    }
}

fn build_inline_chunks_3(
    points: &[f64],
    weights: Option<&[f64]>,
    point_count: usize,
    chunk_capacity: f64,
) -> Vec<[f32; 4]> {
    let mut chunks = Vec::with_capacity(point_count);
    for (row, px) in points.as_chunks::<3>().0.iter().enumerate() {
        let p0 = px[0] as f32;
        let p1 = px[1] as f32;
        let p2 = px[2] as f32;
        let mut remaining = weights.map_or(1.0, |all| all[row]);
        while remaining > chunk_capacity {
            chunks.push([p0, p1, p2, chunk_capacity as f32]);
            remaining -= chunk_capacity;
        }
        if remaining > 0.0 {
            chunks.push([p0, p1, p2, remaining as f32]);
        }
    }
    chunks
}

fn build_inline_chunks_4(
    points: &[f64],
    weights: Option<&[f64]>,
    point_count: usize,
    chunk_capacity: f64,
) -> Vec<[f32; 5]> {
    let mut chunks = Vec::with_capacity(point_count);
    for (row, px) in points.as_chunks::<4>().0.iter().enumerate() {
        let p0 = px[0] as f32;
        let p1 = px[1] as f32;
        let p2 = px[2] as f32;
        let p3 = px[3] as f32;
        let mut remaining = weights.map_or(1.0, |all| all[row]);
        while remaining > chunk_capacity {
            chunks.push([p0, p1, p2, p3, chunk_capacity as f32]);
            remaining -= chunk_capacity;
        }
        if remaining > 0.0 {
            chunks.push([p0, p1, p2, p3, remaining as f32]);
        }
    }
    chunks
}

/// Split rows into shuffled chunks, splitting heavy rows so no chunk is larger
/// than `chunk_capacity`.
fn build_chunks(
    weights: Option<&[f64]>,
    point_count: usize,
    chunk_capacity: f64,
) -> Vec<(u32, f64)> {
    debug_assert!(u32::try_from(point_count).is_ok());
    let mut chunks = Vec::with_capacity(point_count);
    for row in 0..point_count {
        let mut remaining = weights.map_or(1.0, |all| all[row]);
        let row_id = row as u32;
        while remaining > chunk_capacity {
            chunks.push((row_id, chunk_capacity));
            remaining -= chunk_capacity;
        }
        if remaining > 0.0 {
            chunks.push((row_id, remaining));
        }
    }
    chunks
}

/// Replace each non-empty position with its mean over the final pass.
fn finalize_centroids(
    pass_counts: &[f64],
    pass_sums: &[f64],
    centroids: &mut [f64],
    dimensions: usize,
) {
    pass_counts.iter().enumerate().for_each(|(cluster, count)| {
        if *count > 0.0 {
            let base = cluster * dimensions;
            for part in 0..dimensions {
                centroids[base + part] = pass_sums[base + part] / *count;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::laws::round_to_power_of_two;
    use super::*;
    use crate::kmeans_triangle::get_distance_squared;
    use fearless_simd::Level;
    use laws::{resolve_alpha0, resolve_batch_size};

    fn points(count: usize, dimensions: usize) -> Vec<f64> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        (0..count * dimensions)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as f64 / 512.0
            })
            .collect()
    }

    #[test]
    fn the_batch_law_rounds_to_a_power_of_two_and_caps_at_half_the_data() {
        let batch = resolve_batch_size(1_000_000.0, 32, 5, 3, None);
        assert!(batch.is_power_of_two(), "{batch}");
        assert!(batch <= 500_000, "{batch}");

        let small = resolve_batch_size(8.0, 4, 5, 3, None);
        assert!((1..=8).contains(&small), "{small}");

        assert_eq!(resolve_batch_size(1_000.0, 4, 5, 3, Some(800)), 800);
        assert_eq!(resolve_batch_size(10.0, 4, 5, 3, Some(800)), 10);
    }

    #[test]
    fn the_batch_rounding_uses_the_nearest_power_of_two() {
        assert_eq!(round_to_power_of_two(0.5), 1);
        assert_eq!(round_to_power_of_two(1.0), 1);
        assert_eq!(round_to_power_of_two(2.0), 2);
        assert_eq!(round_to_power_of_two(3.0), 4);
        assert_eq!(round_to_power_of_two(4.0), 4);
        assert_eq!(round_to_power_of_two(5.0), 4);
        assert_eq!(round_to_power_of_two(6.0), 8);
        assert_eq!(round_to_power_of_two(1_000_000.0), 1_048_576);
        assert_eq!(round_to_power_of_two(f64::INFINITY), 1);
    }

    #[test]
    fn the_step_law_is_clamped() {
        assert!((1e-4..=1.0).contains(&resolve_alpha0(1.0, 2, 1, 1)));
        assert!((1e-4..=1.0).contains(&resolve_alpha0(1e12, 1_000_000, 25, 1)));
        assert!((1e-4..=1.0).contains(&resolve_alpha0(1_000_000.0, 16, 5, 4_096)));
    }

    #[test]
    fn the_step_law_respects_explicit_batch_size() {
        let small_batch = resolve_alpha0(100_000.0, 16, 5, 256);
        let large_batch = resolve_alpha0(100_000.0, 16, 5, 4_096);
        assert!(
            large_batch > small_batch,
            "expected larger batch to produce larger alpha0: {small_batch} vs {large_batch}"
        );
    }

    #[test]
    fn revive_unused_preserves_donor_mass_and_avoids_reusing_donors_or_dead_clusters() {
        let mut centroids = vec![
            10.0, 10.0, 10.0, // cluster 0 (dead)
            20.0, 20.0, 20.0, // cluster 1 (dead)
            100.0, 100.0, 100.0, // cluster 2 (healthy)
            200.0, 200.0, 200.0, // cluster 3 (healthy)
        ];
        let mut usage = vec![1.0, 2.0, 500.0, 500.0];
        let total_healthy = 1000.0;
        let mut random = Scramble::from_random(&mut Random::from_seed(42));

        let splits = revive_unused(
            &mut centroids,
            &mut usage,
            3,
            REVIVAL_THRESHOLD,
            &mut random,
        );

        assert_eq!(splits, 2);
        assert_eq!(usage.iter().sum::<f64>(), total_healthy);
        for (idx, mass) in usage.iter().enumerate() {
            assert_eq!(
                *mass, 250.0,
                "cluster {idx} should hold half of a donor's 500.0 mass, got {usage:?}"
            );
        }
        let rows: Vec<&[f64]> = centroids
            .as_chunks::<3>()
            .0
            .iter()
            .map(|row| row.as_slice())
            .collect();
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                assert_ne!(rows[i], rows[j], "clusters {i} and {j} collided");
            }
        }
    }

    #[test]
    fn paired_scans_choose_the_same_integer_centroid_as_a_linear_scan() {
        let level = Level::baseline();
        dispatch!(level, simd => {
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % 256) as f64
            };

            for dimensions in 1..=8 {
                for k in 2usize..=17 {
                    let centroids: Vec<f64> = (0..k * dimensions).map(|_| next()).collect();
                    let k_pairs = k.div_ceil(2);
                    let mut panels_f64 = vec![f64::INFINITY; k_pairs * dimensions * 2];
                    pack_centroids_f64_panels(&centroids, dimensions, k, &mut panels_f64);

                    let k_panels = k.div_ceil(4);
                    let mut panels_f32 = vec![f32::INFINITY; k_panels * dimensions * 4];
                    pack_centroids_f32_panels(&centroids, dimensions, &mut panels_f32);

                    let dot_panels_3 = if dimensions == 3 {
                        let mut p = vec![f32::INFINITY; k_panels * 16];
                        pack_centroids_f32_dot_panels(&centroids, 3, &mut p);
                        p
                    } else {
                        Vec::new()
                    };
                    let dot_panels_4 = if dimensions == 4 {
                        let mut p = vec![f32::INFINITY; k_panels * 20];
                        pack_centroids_f32_dot_panels(&centroids, 4, &mut p);
                        p
                    } else {
                        Vec::new()
                    };

                    for _ in 0..16 {
                        let batch_pts: Vec<Vec<f64>> = (0..4)
                            .map(|_| (0..dimensions).map(|_| next()).collect())
                            .collect();
                        let batch_pts_f32: Vec<Vec<f32>> = batch_pts
                            .iter()
                            .map(|p| p.iter().map(|v| *v as f32).collect())
                            .collect();

                        let mut expected_batch = [0usize; 4];
                        for m in 0..4 {
                            let point = &batch_pts[m];
                            let mut expected = 0usize;
                            let mut best = f64::MAX;
                            for cluster in 0..k {
                                let distance = get_distance_squared(
                                    simd,
                                    point,
                                    &centroids,
                                    dimensions,
                                    cluster * dimensions,
                                );
                                if distance < best {
                                    best = distance;
                                    expected = cluster;
                                }
                            }
                            expected_batch[m] = expected;

                            let f64_choice = nearest_f64_panel(
                                simd,
                                point,
                                &panels_f64,
                                dimensions,
                                k_pairs,
                            );
                            let f32_choice = match dimensions {
                                3 => nearest_f32_3_dot(
                                    simd,
                                    &batch_pts_f32[m],
                                    &dot_panels_3,
                                    k_panels,
                                ),
                                4 => nearest_f32_4_dot(
                                    simd,
                                    &batch_pts_f32[m],
                                    &dot_panels_4,
                                    k_panels,
                                ),
                                _ => nearest_centroid_f32_generic(
                                    simd,
                                    &batch_pts_f32[m],
                                    &panels_f32,
                                    dimensions,
                                    k_panels,
                                ),
                            };
                            assert_eq!(f64_choice, expected, "f64 panel d={dimensions} k={k}");
                            assert_eq!(f32_choice, expected, "f32 d={dimensions} k={k}");
                        }

                        let got4_f64 = nearest_4x_f64_generic(
                            simd,
                            [
                                &batch_pts[0],
                                &batch_pts[1],
                                &batch_pts[2],
                                &batch_pts[3],
                            ],
                            &panels_f64,
                            dimensions,
                            k_pairs,
                        );
                        assert_eq!(got4_f64, expected_batch, "4x f64 d={dimensions} k={k}");

                        if dimensions == 3 {
                            let got4 = nearest_4x_f32_3_dot(
                                simd,
                                [
                                    [batch_pts_f32[0][0], batch_pts_f32[0][1], batch_pts_f32[0][2]],
                                    [batch_pts_f32[1][0], batch_pts_f32[1][1], batch_pts_f32[1][2]],
                                    [batch_pts_f32[2][0], batch_pts_f32[2][1], batch_pts_f32[2][2]],
                                    [batch_pts_f32[3][0], batch_pts_f32[3][1], batch_pts_f32[3][2]],
                                ],
                                &dot_panels_3,
                                k_panels,
                            );
                            assert_eq!(got4, expected_batch, "4x3d dot d={dimensions} k={k}");
                        } else if dimensions == 4 {
                            let got4 = nearest_4x_f32_4_dot(
                                simd,
                                [
                                    [
                                        batch_pts_f32[0][0],
                                        batch_pts_f32[0][1],
                                        batch_pts_f32[0][2],
                                        batch_pts_f32[0][3],
                                    ],
                                    [
                                        batch_pts_f32[1][0],
                                        batch_pts_f32[1][1],
                                        batch_pts_f32[1][2],
                                        batch_pts_f32[1][3],
                                    ],
                                    [
                                        batch_pts_f32[2][0],
                                        batch_pts_f32[2][1],
                                        batch_pts_f32[2][2],
                                        batch_pts_f32[2][3],
                                    ],
                                    [
                                        batch_pts_f32[3][0],
                                        batch_pts_f32[3][1],
                                        batch_pts_f32[3][2],
                                        batch_pts_f32[3][3],
                                    ],
                                ],
                                &dot_panels_4,
                                k_panels,
                            );
                            assert_eq!(got4, expected_batch, "4x4d dot d={dimensions} k={k}");
                        }

                        let got4_gen = nearest_4x_f32_generic(
                            simd,
                            [
                                &batch_pts_f32[0],
                                &batch_pts_f32[1],
                                &batch_pts_f32[2],
                                &batch_pts_f32[3],
                            ],
                            &panels_f32,
                            dimensions,
                            k_panels,
                        );
                        assert_eq!(got4_gen, expected_batch, "4x generic d={dimensions} k={k}");
                    }
                }
            }
        });
    }

    #[test]
    fn clusters_two_well_separated_groups() {
        let mut data = Vec::new();
        for index in 0..200 {
            data.extend_from_slice(&[index as f64 * 0.001, 0.0]);
        }
        for index in 0..200 {
            data.extend_from_slice(&[100.0 + index as f64 * 0.001, 0.0]);
        }

        let result = emeans_dispatched(2, 5, &data, 2, EmeansOptions::default());

        assert_eq!(result.centroids.len(), 4);
        let mut means: Vec<f64> = result
            .centroids
            .as_chunks::<2>()
            .0
            .iter()
            .map(|centroid| centroid[0])
            .collect();
        means.sort_by(f64::total_cmp);
        assert!(means[0] < 1.0, "{means:?}");
        assert!(means[1] > 99.0, "{means:?}");
        assert!(result.centroids.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_weighted_histogram_and_the_pixels_it_stands_for_agree() {
        let colours = [[10.0, 10.0], [10.0, 200.0], [200.0, 10.0], [200.0, 200.0]];
        let weights = [50.0, 20.0, 10.0, 100.0];
        let mut collapsed = Vec::new();
        for colour in colours {
            collapsed.extend_from_slice(&colour);
        }
        let mut expanded = Vec::new();
        for (colour, weight) in colours.iter().zip(weights.iter()) {
            for _ in 0..(*weight as usize) {
                expanded.extend_from_slice(colour);
            }
        }
        let start = vec![
            10.0, 10.0, //
            10.0, 200.0, //
            200.0, 10.0, //
            200.0, 200.0,
        ];

        let weighted = emeans_dispatched(
            4,
            1,
            &collapsed,
            2,
            EmeansOptions {
                weights: Some(&weights),
                initial_centroids: Some(&start),
                batch_size: Some(1000),
                revival_threshold: Some(0.0),
                capture_assignments: false,
                seed: None,
                use_f32_assignment: false,
                use_dot_product: false,
                ..Default::default()
            },
        );
        let plain = emeans_dispatched(
            4,
            1,
            &expanded,
            2,
            EmeansOptions {
                weights: None,
                initial_centroids: Some(&start),
                batch_size: Some(1000),
                revival_threshold: Some(0.0),
                capture_assignments: false,
                seed: None,
                use_f32_assignment: false,
                use_dot_product: false,
                ..Default::default()
            },
        );

        for (left, right) in weighted.centroids.iter().zip(plain.centroids.iter()) {
            assert!((left - right).abs() < 1e-12, "{left} vs {right}");
        }
    }

    #[test]
    fn random_input_produces_finite_centroids_and_assignments() {
        let data = points(500, 7);

        let result = emeans_dispatched(
            8,
            3,
            &data,
            7,
            EmeansOptions {
                capture_assignments: true,
                ..Default::default()
            },
        );

        assert_eq!(result.centroids.len(), 8 * 7);
        assert!(result.centroids.iter().all(|value| value.is_finite()));
        let assignments = result.assignments();
        assert_eq!(assignments.len(), 500);
        assert!(assignments.iter().all(|cluster| *cluster < 8));
        assert!(result.updates > 0);
    }

    #[test]
    fn returns_nothing_for_empty_input() {
        let result = emeans_dispatched(3, 5, &[], 3, EmeansOptions::default());

        assert!(result.centroids.is_empty());
        assert_eq!(result.epochs, 0);
        assert_eq!(result.updates, 0);
        assert_eq!(result.batch_size, 0);
    }

    #[test]
    fn a_single_colour_stays_single_and_finite() {
        let data = vec![10.0, 20.0, 30.0];

        let result = emeans_dispatched(8, 5, &data, 3, EmeansOptions::default());

        assert_eq!(result.centroids.len(), 24);
        assert!(
            result
                .centroids
                .as_chunks::<3>()
                .0
                .iter()
                .all(|centroid| centroid == &[10.0, 20.0, 30.0])
        );
    }
}
