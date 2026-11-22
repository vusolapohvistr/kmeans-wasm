//! e-means clustering: mini-batch k-means with EMA centroids.
//!
//! This follows Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze and Jégou,
//! *e-means: outperforming k-means (in one-tenth the epochs)*, arXiv:2611.09501,
//! and the reference implementation in Faiss `contrib/e_means`.
//!
//! The state is the one the paper describes: a positive mass `N_j` and a
//! position `c_j` per cluster. A batch contributes counts `n_j` and sums `s_j`,
//! and one update is
//!
//! ```text
//! N_j <- gamma * N_j + (1 - gamma) * n_j
//! c_j <- c_j + alpha * n_j / N_j * (s_j / n_j - c_j)
//! ```
//!
//! where `alpha = 1 - gamma` is cosine-annealed from `alpha0` to `alpha0 / 10`.
//! Batch size and `alpha0` come from the paper's scaling laws, so the only
//! caller-facing choice is the number of passes.
//!
//! The module is split into smaller pure pieces: assignment scans, update
//! rules, laws/schedule helpers, and entropy/shuffle utilities. The e-means
//! core remains the only hot path and keeps its SIMD-heavy loops in
//! `assignment::nearest_centroid_fixed`.

mod assignment;
mod laws;
mod update;
mod util;

use assignment::{nearest_centroid, nearest_centroid_fixed};
use laws::{
    batch_size_from_raw, resolve_alpha0_from_n_k_epochs, resolve_batch_raw, scheduled_alpha,
};
use update::{apply_batch, revive_unused, transpose_centroids};
use util::{Scramble, shuffle};

use fearless_simd::{Level, Simd, dispatch};

use crate::kmeans_triangle::{Random, get_centroids_with_random};

/// Passes over the data when the caller does not choose. The paper's reference
/// implementation found five enough to beat Lloyd's converged loss on its
/// benchmarks, and more passes stopped improving in practice.
pub const DEFAULT_EPOCHS: usize = 5;

/// Revive under-used clusters, measured in batch updates.
const CHECK_REVIVAL_EVERY: usize = 5;

/// Initial mass. The paper uses `1e-10`; the reference implementation uses
/// `2^-16` to keep the first batch decisive without making the subtraction of
/// old mass observable. Either is negligible against a batch, so this follows
/// the paper.
const INIT_USAGE: f64 = 1e-10;

/// A cluster below this fraction of the mean mass is revived.
const REVIVAL_THRESHOLD: f64 = 0.1;

/// Revival hurt held-out error at two dimensions in the reference
/// implementation, and two dimensions is exactly the case where a split is
/// most likely to cut a thin cluster in half. Above it, revival is the single
/// largest contributor to e-means' quality.
const NO_REVIVAL_MAX_DIMENSIONS: usize = 2;

/// Split a heavy row so no chunk is larger than this fraction of a batch. The
/// exact value trades assignment work against how evenly a dominant colour is
/// spread through the shuffled chunk stream.
const CHUNKS_PER_BATCH: f64 = 16.0;

/// Optional inputs to the core. The public packed and general entry points use
/// the defaults; the fields exist so tests can pin the start and so the
/// weighted packed path can pass a histogram.
#[derive(Clone, Copy, Default)]
pub struct EmeansOptions<'a> {
    /// A weight per row, for input where many points are exactly equal.
    ///
    /// Must be `None` or the same length as `points` divided by `dimensions`.
    pub weights: Option<&'a [f64]>,

    /// Replaces the random draw, for comparing runs from one start.
    pub initial_centroids: Option<&'a [f64]>,

    /// Overrides the scaling law. Clamped to the number of points.
    pub batch_size: Option<usize>,

    /// Overrides the dimension-dependent revival threshold. Zero disables revival.
    pub revival_threshold: Option<f64>,

    /// Records the last-epoch assignment of every row. The general API needs it;
    /// the packed paths do not.
    pub capture_assignments: bool,
}

/// The outcome of one e-means run.
pub struct EmeansResult {
    /// Flat row-major centroids after last-epoch-mean finalization, `k * dimensions`.
    pub centroids: Vec<f64>,
    pub dimensions: usize,
    /// Completed passes over the data.
    pub epochs: usize,
    /// Batch updates applied. Useful for tests and for reporting the recipe.
    pub updates: usize,
    /// The batch size chosen by the law or supplied by the caller.
    pub batch_size: usize,
    /// The initial EMA step chosen by the law.
    pub alpha0: f64,
    /// Last-epoch assignment per row, when requested.
    pub assignments: Option<Vec<u32>>,
}

impl EmeansResult {
    /// One array per cluster, as the general `emeans` API returns it.
    pub fn centroid_rows(&self) -> Vec<Vec<f64>> {
        self.centroids
            .chunks_exact(self.dimensions)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// The assignment of every row, or an empty slice when not captured.
    pub fn assignments(&self) -> &[u32] {
        self.assignments.as_deref().unwrap_or(&[])
    }
}

/// Runs e-means once at the strongest SIMD level this build supports.
///
/// As with the Hamerly core, the level is a compile-time constant, so this
/// selects it once per run rather than once per distance.
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
///
/// `dimensions` must be non-zero and `points.len()` a multiple of it. `k` may
/// exceed the number of rows, in which case the initial draw repeats its last
/// pick; the result still has `k` centroids.
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

    let point_total = total_weight.ceil().min(usize::MAX as f64) as usize;
    let batch_size = if let Some(given) = options.batch_size {
        given.clamp(1, point_total)
    } else {
        let raw = resolve_batch_raw(total_weight, k, epochs, dimensions, None);
        batch_size_from_raw(raw, total_weight, dimensions)
    };
    let alpha0 = resolve_alpha0_from_n_k_epochs(total_weight, k, epochs, dimensions);
    let revival_threshold = options.revival_threshold.unwrap_or(
        if dimensions <= NO_REVIVAL_MAX_DIMENSIONS || k > point_count {
            0.0
        } else {
            REVIVAL_THRESHOLD
        },
    );

    let mut entropy = Random::new();
    let mut centroids = match options.initial_centroids {
        Some(given) => {
            debug_assert_eq!(given.len(), k * dimensions);
            given.to_vec()
        }
        None => {
            get_centroids_with_random(points, dimensions, point_count, k, weights, &mut entropy)
        }
    };
    // The initial draw stays on the same entropy source as the Lloyd path, so a
    // patched `Math.random` gives both solvers the same starting centroids. The
    // rest of the run needs millions of draws for shuffles and revival, and a
    // call into JavaScript for each one costs more than the shuffle itself. Seed
    // a local stream once; reproducibility is unchanged because the seed still
    // comes from the caller-visible entropy.
    let mut random = Scramble::from_random(&mut entropy);
    let mut usage = vec![INIT_USAGE; k];

    let batches_per_epoch = (total_weight / batch_size as f64).ceil().max(1.0) as usize;
    let total_steps = epochs.saturating_mul(batches_per_epoch);
    let mut updates = 0usize;

    let batch_capacity = batch_size as f64;
    // A distinct colour can stand for tens of thousands of pixels. Emitting all
    // of its weight at one point would make the batches look like consecutive
    // blocks of one colour, which is not a shuffle of the pixels it represents.
    // Split large weights into chunks, shuffle those, and fill batches from the
    // shuffled stream. Small weights, which are almost every colour in a
    // photograph, are unchanged.
    let chunk_capacity = (batch_capacity / CHUNKS_PER_BATCH).max(1.0);
    // The chunks depend only on the weights, so build the multiset once and
    // reshuffle it for each epoch. Shuffling an already-shuffled sequence is
    // still uniform, and rebuilding it every epoch only repeated 5n pushes.
    let mut chunks = build_chunks(weights, point_count, chunk_capacity);
    let mut assignments = options.capture_assignments.then(|| vec![0u32; point_count]);
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
    let mut transposed = vec![0.0f64; dimensions * k];
    transpose_centroids(&centroids, dimensions, k, &mut transposed);
    // Counts and sums share one allocation; `apply_batch` borrows the two halves
    // as separate slices.
    let mut batch_stats = vec![0.0f64; k + k * dimensions];
    let (batch_counts, batch_sums) = batch_stats.split_at_mut(k);
    let mut touched = Vec::with_capacity(k.min(point_count));
    let mut touched_flags = vec![false; k];

    // Last-epoch-mean finalization: accumulate the assignments made during the
    // final pass, then replace each non-empty position by that mean. A cluster
    // that receives nothing keeps its current position rather than becoming NaN.
    let mut pass_stats = vec![0.0f64; k + k * dimensions];
    let (pass_counts, pass_sums) = pass_stats.split_at_mut(k);

    for epoch in 0..epochs {
        shuffle(&mut chunks, &mut random);
        let last_epoch = epoch + 1 == epochs;
        let mut cursor = 0usize;
        let mut remaining = chunks[0].1;

        while cursor < chunks.len() {
            let mut batch_weight = 0.0f64;

            while cursor < chunks.len() && batch_weight < batch_capacity {
                let row = chunks[cursor].0;
                let take = remaining.min(batch_capacity - batch_weight);
                let point_offset = row * dimensions;
                let point = &points[point_offset..point_offset + dimensions];
                let centroid = match dimensions {
                    3 => nearest_centroid_fixed::<3, _>(simd, point, &centroids, &transposed, k),
                    4 => nearest_centroid_fixed::<4, _>(simd, point, &centroids, &transposed, k),
                    _ => nearest_centroid(simd, point, &centroids, &transposed, dimensions, k),
                };

                if !touched_flags[centroid] {
                    touched_flags[centroid] = true;
                    touched.push(centroid);
                }
                batch_counts[centroid] += take;
                let sum_offset = centroid * dimensions;
                for part in 0..dimensions {
                    let contribution = take * points[point_offset + part];
                    batch_sums[sum_offset + part] += contribution;
                    if last_epoch {
                        pass_sums[sum_offset + part] += contribution;
                    }
                }
                if last_epoch {
                    pass_counts[centroid] += take;
                    if let Some(ref mut labels) = assignments {
                        labels[row] = centroid as u32;
                    }
                }

                batch_weight += take;
                remaining -= take;
                if remaining <= 0.0 {
                    cursor += 1;
                    if cursor < chunks.len() {
                        remaining = chunks[cursor].1;
                    }
                }
            }

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
            transpose_centroids(&centroids, dimensions, k, &mut transposed);
            updates += 1;

            for &centroid in &touched {
                batch_counts[centroid] = 0.0;
                let base = centroid * dimensions;
                batch_sums[base..base + dimensions].fill(0.0);
                touched_flags[centroid] = false;
            }
            touched.clear();

            if updates.is_multiple_of(CHECK_REVIVAL_EVERY) {
                let splits = revive_unused(
                    &mut centroids,
                    &mut usage,
                    dimensions,
                    revival_threshold,
                    &mut random,
                );
                if splits > 0 {
                    transpose_centroids(&centroids, dimensions, k, &mut transposed);
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

/// Split rows into shuffled chunks, splitting heavy rows so no chunk is larger
/// than `chunk_capacity`.
fn build_chunks(
    weights: Option<&[f64]>,
    point_count: usize,
    chunk_capacity: f64,
) -> Vec<(usize, f64)> {
    let mut chunks = Vec::with_capacity(point_count);
    for row in 0..point_count {
        let mut remaining = weights.map_or(1.0, |all| all[row]);
        while remaining > chunk_capacity {
            chunks.push((row, chunk_capacity));
            remaining -= chunk_capacity;
        }
        if remaining > 0.0 {
            chunks.push((row, remaining));
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
    use fearless_simd::Level;
    use laws::{resolve_alpha0_from_n_k_epochs, resolve_batch_size};

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
        assert!((1e-4..=1.0).contains(&resolve_alpha0_from_n_k_epochs(1.0, 2, 1, 3)));
        assert!((1e-4..=1.0).contains(&resolve_alpha0_from_n_k_epochs(1e12, 1_000_000, 25, 3)));
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
                for k in 2..=9 {
                    let centroids: Vec<f64> = (0..k * dimensions).map(|_| next()).collect();
                    let mut transposed = vec![0.0; dimensions * k];
                    transpose_centroids(&centroids, dimensions, k, &mut transposed);

                    for _ in 0..50 {
                        let point: Vec<f64> = (0..dimensions).map(|_| next()).collect();
                        let expected = nearest_centroid(
                            simd,
                            &point,
                            &centroids,
                            &transposed,
                            dimensions,
                            k,
                        );
                        let generic = nearest_centroid(
                            simd,
                            &point,
                            &centroids,
                            &transposed,
                            dimensions,
                            k,
                        );
                        let fixed = match dimensions {
                            3 => nearest_centroid_fixed::<3, _>(
                                simd,
                                &point,
                                &centroids,
                                &transposed,
                                k,
                            ),
                            4 => nearest_centroid_fixed::<4, _>(
                                simd,
                                &point,
                                &centroids,
                                &transposed,
                                k,
                            ),
                            _ => generic,
                        };
                        assert_eq!(generic, expected, "d={dimensions} k={k}");
                        assert_eq!(fixed, expected, "d={dimensions} k={k}");
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
