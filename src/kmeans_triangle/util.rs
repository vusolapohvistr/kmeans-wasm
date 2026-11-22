//! Shared utilities for the Hamerly core: the distance kernel, entropy source,
//! and initial centroid draw.

use fearless_simd::{Simd, SimdBase, f64x2};

/// Entropy source for the initial centroid draw.
///
/// The clustering algorithm below is shared, and only the random stream differs
/// per target. On WebAssembly it has to come from `Math.random`, because that is
/// what callers replace to make a run reproducible, for example the comparison
/// page seeds it before every measurement. Natively it is a seeded stream so the
/// test suite and the benchmarks stay repeatable.
#[cfg(target_arch = "wasm32")]
pub struct Random;

#[cfg(target_arch = "wasm32")]
impl Random {
    pub fn new() -> Self {
        Random
    }

    pub fn next(&mut self) -> f64 {
        js_sys::Math::random()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Random(rand::rngs::SmallRng);

#[cfg(not(target_arch = "wasm32"))]
impl Random {
    pub fn new() -> Self {
        use rand::SeedableRng;

        // Fixed seed so the native test suite and the benchmarks stay repeatable.
        Random(rand::rngs::SmallRng::seed_from_u64(0))
    }

    pub fn next(&mut self) -> f64 {
        rand::RngExt::random(&mut self.0)
    }
}

/// Attempts to spend finding a value that is not already a centroid, per pick.
const MAX_DRAWS_PER_PICK: usize = 64;

/// Picks the initial centroids as `k` distinct input points.
pub fn get_centroids(
    points: &[f64],
    dimensions: usize,
    point_count: usize,
    k: usize,
    weights: Option<&[f64]>,
) -> Vec<f64> {
    let mut random = Random::new();
    get_centroids_with_random(points, dimensions, point_count, k, weights, &mut random)
}

/// As `get_centroids`, but consumes draws from a caller-owned stream.
///
/// e-means needs one stream for both the initial draw and the epoch shuffles.
/// Hamerly has no other draws, so its wrapper above keeps the old sequence and
/// the recorded fingerprints are unaffected.
pub fn get_centroids_with_random(
    points: &[f64],
    dimensions: usize,
    point_count: usize,
    k: usize,
    weights: Option<&[f64]>,
    random: &mut Random,
) -> Vec<f64> {
    let distinct = k.min(point_count);

    // Sampling with probability proportional to weight. When the caller has
    // collapsed equal points into one weighted entry this keeps the draw
    // distributed exactly as it would be over the original pixels, so collapsing
    // does not quietly change which centroids the run starts from.
    let cumulative: Option<Vec<f64>> = weights.map(|all| {
        let mut running = Vec::with_capacity(point_count);
        let mut total = 0.0;
        for weight in all.iter().take(point_count) {
            total += *weight;
            running.push(total);
        }
        running
    });

    let mut chosen: Vec<usize> = Vec::with_capacity(distinct);
    let mut values: Vec<&[f64]> = Vec::with_capacity(distinct);

    // Reject a draw whose value is already a centroid. Equal values would seed
    // two coincident centroids, one of which then wins every point that both
    // could have taken and the other is starved, so a single starved cluster is
    // enough to leave a palette entry no pixel can reach.
    for _ in 0..distinct {
        let mut candidate = 0;
        // Rejection slows down as the pool of unused values runs out. When k is
        // close to the number of distinct values there is nothing to reject and
        // the draw is taken as it comes, so cap the work per pick.
        for _ in 0..MAX_DRAWS_PER_PICK {
            candidate = match cumulative {
                Some(ref running) => {
                    let total = *running.last().unwrap_or(&1.0);
                    let target = random.next() * total;
                    match running.binary_search_by(|value| value.total_cmp(&target)) {
                        Ok(index) | Err(index) => index.min(point_count - 1),
                    }
                }
                None => ((random.next() * (point_count as f64)) as usize).min(point_count - 1),
            };
            let value = &points[candidate * dimensions..(candidate + 1) * dimensions];
            if !values.contains(&value) {
                break;
            }
        }
        let value = &points[candidate * dimensions..(candidate + 1) * dimensions];
        values.push(value);
        chosen.push(candidate);
    }

    // When k is above the number of input points there is nothing left to draw,
    // so the remainder repeats the last pick rather than rejecting forever.
    chosen.resize(k, *chosen.last().expect("point_count is not zero"));

    let mut centroids = Vec::with_capacity(k * dimensions);
    for index in chosen {
        let start = index * dimensions;
        centroids.extend_from_slice(&points[start..start + dimensions]);
    }

    centroids
}

#[inline(always)]
pub fn get_distance_squared<S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    dimensions: usize,
    centroid_offset: usize,
) -> f64 {
    // This is the innermost loop of the whole crate, and it is not bound by
    // floating-point throughput: the scalar form serialises every point through
    // one accumulator, so the add latency chain, not the arithmetic, sets the
    // pace. Two lanes halve that chain and halve the instruction count per
    // point, which is what the four-dimensional and three-dimensional cases
    // live on.
    #[cfg(feature = "counters")]
    crate::work::count_one(&crate::work::DISTANCE_EVALUATIONS);

    let centroid = &centroids[centroid_offset..centroid_offset + dimensions];
    let even = dimensions - (dimensions % 2);
    let mut accumulator = f64x2::<S>::splat(simd, 0.0);

    for (point_pair, centroid_pair) in point[..even]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(centroid[..even].as_chunks::<2>().0.iter())
    {
        let difference = f64x2::<S>::from_slice(simd, point_pair.as_slice())
            - f64x2::<S>::from_slice(simd, centroid_pair.as_slice());
        accumulator += difference * difference;
    }

    let mut sum = accumulator[0] + accumulator[1];
    for (point_value, centroid_value) in point[even..].iter().zip(centroid[even..].iter()) {
        let difference = point_value - centroid_value;
        sum += difference * difference;
    }

    sum
}
