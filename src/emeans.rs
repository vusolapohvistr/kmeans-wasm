//! e-means clustering: mini-batch k-means with EMA centroids.
//!
//! This follows Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze and Jégou,
//! *e-means: outperforming k-means (in one-tenth the epochs)*, arXiv:2611.09501,
//! and the reference implementation in Faiss `contrib/e_means`.
//!
//! The state is the one the paper describes: a positive mass `N_j` and a
//! position `c_j` per cluster. A batch contributes counts `n_j` and sums `s_j`,
//! and the mass and position move by
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
//! This is deliberately a separate module from `kmeans_triangle`: the shipped
//! Hamerly path has a bit-for-bit output invariant, while e-means is a
//! stochastic mini-batch method. Keeping them apart means neither can quietly
//! change the other's arithmetic.
//!
//! Points may be weighted. The packed paths collapse equal colours to one row
//! with a weight, which is exact for Lloyd because equal points are
//! interchangeable. Here it is also what makes e-means affordable on images: a
//! batch is filled by total weight, so a distinct row can be assigned once and
//! contribute all of its pixels. The resulting order is not the exact order of
//! the expanded pixels, but the input is a multiset and the method is
//! stochastic, so that is an acceptable trade for a reduction of one to two
//! orders of magnitude on real photographs.

use fearless_simd::{Level, Simd, SimdBase, dispatch, f64x2};

use crate::kmeans_triangle::{Random, get_centroids_with_random, get_distance_squared};

/// Passes over the data when the caller does not choose. The paper's reference
/// implementation found five enough to beat Lloyd's converged loss on its
/// benchmarks, and more passes stopped improving in practice.
pub const DEFAULT_EPOCHS: usize = 5;

/// The cosine schedule ends at one tenth of the initial step.
const ALPHA_MIN_FACTOR: f64 = 0.1;

/// Revive under-used clusters, measured in batch updates.
const CHECK_REVIVAL_EVERY: usize = 5;

/// Initial mass. The paper uses `1e-10`; the reference implementation uses
/// `2^-16` to keep the first batch decisive without making the subtraction of
/// old mass observable. Either is negligible against a batch, so this follows
/// the paper.
const INIT_USAGE: f64 = 1e-10;

/// Relative perturbation applied to a donor centroid when it is split.
const SPLIT_EPSILON: f64 = 1e-4;

/// A cluster below this fraction of the mean mass is revived.
const REVIVAL_THRESHOLD: f64 = 0.1;

/// Revival hurt held-out error at two dimensions in the reference
/// implementation, and two dimensions is exactly the case where a split is
/// most likely to cut a thin cluster in half. Above it, revival is the single
/// largest contributor to e-means' quality.
const NO_REVIVAL_MAX_DIMENSIONS: usize = 2;

/// At low dimension the smaller "quality" batch is worth its extra updates.
/// Above this the throughput batch is the default.
const QUALITY_BATCH_MAX_DIMENSIONS: usize = 16;

/// Split a heavy row so no chunk is larger than this fraction of a batch. The
/// exact value trades assignment work against how evenly a dominant colour is
/// spread through the shuffled chunk stream.
const CHUNKS_PER_BATCH: f64 = 16.0;

// Batch law B = C * k^0.134749 * n^0.876473 * T^0.813344. The quality
// coefficient is one eighth of the throughput one, matching the reference
// implementation's shipped fits (which differ slightly from the paper's
// rounded table values).
const BATCH_QUALITY_COEFFICIENT: f64 = 0.004_391_75;
const BATCH_THROUGHPUT_COEFFICIENT: f64 = 0.035_134;
const BATCH_K_EXPONENT: f64 = 0.134_749;
const BATCH_N_EXPONENT: f64 = 0.876_473;
const BATCH_T_EXPONENT: f64 = 0.813_344;

// Step law alpha0 = C * k^-0.4617 * B^0.9003 * n^-0.4498 * T^-0.1057.
const ALPHA_COEFFICIENT: f64 = 0.824_4;
const ALPHA_K_EXPONENT: f64 = -0.461_7;
const ALPHA_B_EXPONENT: f64 = 0.900_3;
const ALPHA_N_EXPONENT: f64 = -0.449_8;
const ALPHA_T_EXPONENT: f64 = -0.105_7;
const ALPHA_MINIMUM: f64 = 1e-4;

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

    let batch_size = resolve_batch_size(total_weight, k, epochs, dimensions, options.batch_size);
    let alpha0 = resolve_alpha0(total_weight, k, epochs, batch_size);
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
    let mut chunks: Vec<(usize, f64)> = Vec::with_capacity(point_count);
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
    let mut batch_counts = vec![0.0f64; k];
    let mut batch_sums = vec![0.0f64; k * dimensions];
    let mut touched = Vec::with_capacity(k.min(point_count));
    let mut touched_flags = vec![false; k];

    // Last-epoch-mean finalization: accumulate the assignments made during the
    // final pass, then replace each non-empty position by that mean. A cluster
    // that receives nothing keeps its current position rather than becoming NaN.
    let mut pass_counts = vec![0.0f64; k];
    let mut pass_sums = vec![0.0f64; k * dimensions];

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
                &batch_counts,
                &batch_sums,
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

    for (cluster, count) in pass_counts.iter().enumerate() {
        if *count > 0.0 {
            let base = cluster * dimensions;
            let count = *count;
            for part in 0..dimensions {
                centroids[base + part] = pass_sums[base + part] / count;
            }
        }
    }

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

/// The batch size law. `n` is the number of points, not rows, so a collapsed
/// histogram uses the same statistical batch as the pixels it stands for.
fn resolve_batch_size(
    n: f64,
    k: usize,
    epochs: usize,
    dimensions: usize,
    given: Option<usize>,
) -> usize {
    let n = n.max(1.0);
    let point_total = n.ceil().min(usize::MAX as f64) as usize;

    if let Some(batch_size) = given {
        // The batch is measured in points, not rows, so a collapsed histogram
        // may still fill a batch larger than its row count from weighted rows.
        return batch_size.clamp(1, point_total);
    }

    let coefficient = if dimensions <= QUALITY_BATCH_MAX_DIMENSIONS {
        BATCH_QUALITY_COEFFICIENT
    } else {
        BATCH_THROUGHPUT_COEFFICIENT
    };
    let raw = coefficient
        * (k.max(1) as f64).powf(BATCH_K_EXPONENT)
        * n.powf(BATCH_N_EXPONENT)
        * (epochs.max(1) as f64).powf(BATCH_T_EXPONENT);
    let rounded = round_to_power_of_two(raw);
    let cap = if dimensions <= QUALITY_BATCH_MAX_DIMENSIONS {
        (n / 2.0) as usize
    } else {
        round_to_power_of_two(n / 2.0)
    };
    rounded.clamp(1, point_total).min(cap.max(1))
}

/// The initial EMA step law, clamped to the range the reference implementation
/// uses so a tiny or enormous problem cannot produce a step that stalls or
/// overshoots.
fn resolve_alpha0(n: f64, k: usize, epochs: usize, batch_size: usize) -> f64 {
    let alpha = ALPHA_COEFFICIENT
        * (k.max(1) as f64).powf(ALPHA_K_EXPONENT)
        * (batch_size.max(1) as f64).powf(ALPHA_B_EXPONENT)
        * n.max(1.0).powf(ALPHA_N_EXPONENT)
        * (epochs.max(1) as f64).powf(ALPHA_T_EXPONENT);
    alpha.clamp(ALPHA_MINIMUM, 1.0)
}

/// Cosine-anneal the step from `alpha0` at the first update to a tenth of it at
/// the last, as in Equation 5 of the paper.
fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = (step.min(total_steps) as f64) / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Nearest centroid in ascending index order with a strict comparison, so a tie
/// resolves to the lowest index. The scan order is the same as the Hamerly core.
///
/// The e-means assignment phase has a different shape from a Lloyd round: it
/// scans the same small set of centroids for a long stream of independently
/// weighted rows, so it can put the two SIMD lanes on two *clusters* instead of
/// two dimensions. That removes the horizontal add from every distance and
/// halves the loop bookkeeping. `transposed` holds the dimension-major copy that
/// makes the two cluster positions contiguous; it is rebuilt after each batch,
/// which is `k * dimensions` copies against `n * k * dimensions` arithmetic.
#[inline]
fn nearest_centroid<S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    transposed: &[f64],
    dimensions: usize,
    k: usize,
) -> usize {
    let mut nearest = 0;
    let mut nearest_distance_squared = f64::MAX;

    for pair in 0..k / 2 {
        let mut accumulator = f64x2::<S>::splat(simd, 0.0);
        let first_cluster = pair * 2;
        for part in 0..dimensions {
            let offset = part * k + first_cluster;
            let centroid = f64x2::<S>::from_slice(simd, &transposed[offset..offset + 2]);
            let point = f64x2::<S>::splat(simd, point[part]);
            let difference = point - centroid;
            accumulator += difference * difference;
        }

        let first_distance = accumulator[0];
        if first_distance < nearest_distance_squared {
            nearest_distance_squared = first_distance;
            nearest = first_cluster;
        }
        let second_distance = accumulator[1];
        if second_distance < nearest_distance_squared {
            nearest_distance_squared = second_distance;
            nearest = first_cluster + 1;
        }
    }

    if !k.is_multiple_of(2) {
        let distance_squared =
            get_distance_squared(simd, point, centroids, dimensions, (k - 1) * dimensions);
        if distance_squared < nearest_distance_squared {
            nearest = k - 1;
        }
    }

    nearest
}

/// The same scan with the dimension count known at compile time. The packed
/// paths are three and four components, and fixing the loop bound lets the
/// compiler unroll it and drop the per-part multiply by `dimensions`.
#[inline]
fn nearest_centroid_fixed<const D: usize, S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    transposed: &[f64],
    k: usize,
) -> usize {
    let mut nearest = 0;
    let mut nearest_distance_squared = f64::MAX;

    for pair in 0..k / 2 {
        let mut accumulator = f64x2::<S>::splat(simd, 0.0);
        let first_cluster = pair * 2;
        for part in 0..D {
            let offset = part * k + first_cluster;
            let centroid = f64x2::<S>::from_slice(simd, &transposed[offset..offset + 2]);
            let point = f64x2::<S>::splat(simd, point[part]);
            let difference = point - centroid;
            accumulator += difference * difference;
        }

        let first_distance = accumulator[0];
        if first_distance < nearest_distance_squared {
            nearest_distance_squared = first_distance;
            nearest = first_cluster;
        }
        let second_distance = accumulator[1];
        if second_distance < nearest_distance_squared {
            nearest_distance_squared = second_distance;
            nearest = first_cluster + 1;
        }
    }

    if !k.is_multiple_of(2) {
        let distance_squared = get_distance_squared(simd, point, centroids, D, (k - 1) * D);
        if distance_squared < nearest_distance_squared {
            nearest = k - 1;
        }
    }

    nearest
}

fn transpose_centroids(centroids: &[f64], dimensions: usize, k: usize, transposed: &mut [f64]) {
    for cluster in 0..k {
        for part in 0..dimensions {
            transposed[part * k + cluster] = centroids[cluster * dimensions + part];
        }
    }
}

/// One line of Equation 2. `usage` decays for every cluster, and only the
/// clusters a batch touched get a new observation folded in. Untouched clusters
/// therefore keep their position and lose mass, which is exactly the EMA.
fn apply_batch(
    centroids: &mut [f64],
    usage: &mut [f64],
    dimensions: usize,
    counts: &[f64],
    sums: &[f64],
    touched: &[usize],
    alpha: f64,
) {
    let decay = 1.0 - alpha;
    for mass in usage.iter_mut() {
        *mass *= decay;
    }

    for &cluster in touched {
        let count = counts[cluster];
        if count <= 0.0 {
            continue;
        }
        let mass = usage[cluster] + alpha * count;
        if mass <= 0.0 {
            continue;
        }
        let move_fraction = alpha * count / mass;
        let base = cluster * dimensions;
        for part in 0..dimensions {
            let mean = sums[base + part] / count;
            let position = &mut centroids[base + part];
            *position += move_fraction * (mean - *position);
        }
        usage[cluster] = mass;
    }
}

/// Split a donor cluster into two slightly separated children, giving one to
/// the under-used slot. The perturbation alternates sign by component, as in
/// Equation 8, so the split plane passes through the donor centroid.
fn revive_unused(
    centroids: &mut [f64],
    usage: &mut [f64],
    dimensions: usize,
    threshold: f64,
    random: &mut Scramble,
) -> usize {
    if threshold <= 0.0 || usage.is_empty() {
        return 0;
    }

    let mean = usage.iter().sum::<f64>() / usage.len() as f64;
    let cutoff = threshold * mean;

    let unused: Vec<usize> = usage
        .iter()
        .enumerate()
        .filter_map(|(cluster, mass)| (*mass < cutoff).then_some(cluster))
        .collect();
    if unused.is_empty() {
        return 0;
    }

    // Weighted sampling without replacement, drawn by inverse CDF over the
    // remaining donor mass. This is the same distribution as the
    // Efraimidis-Spirakis `log(u) / weight` keys, but it needs one draw per
    // split rather than one per cluster, and it avoids sorting a key per
    // cluster. A donor chosen for one under-used slot cannot be chosen again,
    // so it is blocked after use.
    let mut blocked = vec![false; usage.len()];
    for &cluster in &unused {
        blocked[cluster] = true;
    }

    let mut splits = 0;
    for &target in &unused {
        let mut total = 0.0;
        for (cluster, mass) in usage.iter().enumerate() {
            if !blocked[cluster] && *mass > 0.0 {
                total += *mass;
            }
        }
        if total <= 0.0 {
            break;
        }

        let pick = random.next() * total;
        let mut cumulative = 0.0;
        let mut donor = None;
        for (cluster, mass) in usage.iter().enumerate() {
            if blocked[cluster] || *mass <= 0.0 {
                continue;
            }
            cumulative += *mass;
            donor = Some(cluster);
            if pick < cumulative {
                break;
            }
        }
        let Some(donor) = donor else {
            break;
        };

        // Block the donor before the next draw so its halved mass is not
        // counted again. The target was already blocked as under-used, so it
        // cannot become a donor for a later slot.
        split_centroid(centroids, dimensions, donor, target);
        let half = usage[donor] / 2.0;
        usage[donor] = half;
        usage[target] = half;
        blocked[donor] = true;
        splits += 1;
    }

    splits
}

fn split_centroid(centroids: &mut [f64], dimensions: usize, donor: usize, target: usize) {
    let donor_base = donor * dimensions;
    let target_base = target * dimensions;
    let mut child = [0.0f64; 16];

    // Dimensions above the stack buffer are far outside the packed paths and
    // rare for e-means, so fall back to a heap row rather than a fixed limit.
    let mut heap;
    let child: &mut [f64] = if dimensions <= child.len() {
        &mut child[..dimensions]
    } else {
        heap = vec![0.0; dimensions];
        &mut heap
    };

    for part in 0..dimensions {
        let sign = if part.is_multiple_of(2) { 1.0 } else { -1.0 };
        let offset = SPLIT_EPSILON * sign;
        let value = centroids[donor_base + part];
        centroids[donor_base + part] = value * (1.0 - offset);
        child[part] = value * (1.0 + offset);
    }

    // A zero or otherwise symmetric coordinate can survive the perturbation
    // unchanged. The children must still be distinct, or the revived cluster is
    // an exact duplicate of its donor.
    if (0..dimensions).all(|part| centroids[donor_base + part] == child[part]) {
        let next = child[0].next_up();
        child[0] = if next.is_finite() {
            next
        } else {
            child[0].next_down()
        };
    }

    centroids[target_base..target_base + dimensions].copy_from_slice(child);
}

fn round_to_power_of_two(value: f64) -> usize {
    if !value.is_finite() || value <= 1.0 {
        return 1;
    }
    // Clearing the mantissa leaves the largest power of two not above `value`.
    // The cast saturates for values beyond `usize`, which cannot occur for a
    // point count but keeps this total.
    let low = f64::from_bits(value.to_bits() & 0x7ff0_0000_0000_0000);
    let high = low * 2.0;
    let rounded = if value / low < high / value {
        low
    } else {
        high
    };
    rounded as usize
}

/// The private stream used after the initial centroids are drawn.
///
/// SplitMix64 is enough here: this is a shuffle, not a simulation, and the only
/// requirement is a uniform-enough permutation from a seed the caller can
/// reproduce. Using it avoids one call into JavaScript per shuffled item on the
/// wasm target, which is a large fixed cost on a million-point pass.
struct Scramble(u64);

impl Scramble {
    fn from_random(random: &mut Random) -> Self {
        let first = random.next().to_bits();
        let second = random.next().to_bits();
        Self(first ^ second.rotate_left(32) ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        // 53 significant bits, so every representable value in [0, 1) is
        // reachable and the conversion is exact.
        (mixed >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn shuffle<T>(items: &mut [T], random: &mut Scramble) {
    for index in (1..items.len()).rev() {
        let other = (random.next() * (index + 1) as f64) as usize;
        let other = other.min(index);
        items.swap(index, other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // A given batch is in points, not rows, so a weighted histogram can
        // ask for a batch larger than its row count.
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
        assert!((ALPHA_MINIMUM..=1.0).contains(&resolve_alpha0(1.0, 2, 1, 1)));
        assert!((ALPHA_MINIMUM..=1.0).contains(&resolve_alpha0(1e12, 1_000_000, 25, 1)));
    }

    #[test]
    fn paired_scans_choose_the_same_integer_centroid_as_a_linear_scan() {
        // Integer coordinates keep every partial sum exact, so reassociating
        // the squares cannot change a distance. That tests the transposed
        // layout and the odd-cluster tail rather than floating-point rounding.
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
                        let mut expected = 0usize;
                        let mut best = f64::MAX;
                        for cluster in 0..k {
                            let distance = get_distance_squared(
                                simd,
                                &point,
                                &centroids,
                                dimensions,
                                cluster * dimensions,
                            );
                            if distance < best {
                                best = distance;
                                expected = cluster;
                            }
                        }

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
    fn the_schedule_starts_at_alpha0_and_ends_near_one_tenth() {
        let alpha0 = 0.4;
        assert_eq!(scheduled_alpha(alpha0, 0, 10), alpha0);
        let last = scheduled_alpha(alpha0, 9, 10);
        assert!(last >= alpha0 * ALPHA_MIN_FACTOR);
        assert!(last < alpha0 * 0.13);
        assert_eq!(scheduled_alpha(alpha0, 10, 10), alpha0 * ALPHA_MIN_FACTOR);
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
        // Four well-separated colours, with unequal multiplicities. With an
        // explicit start and one epoch the weighted run sees each colour once
        // and the expanded run sees the same colour repeated, so the centroids
        // are the same weighted means.
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
