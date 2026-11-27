//! Shared utilities for the Hamerly core: the distance kernel, entropy source,
//! and initial centroid draw.

use fearless_simd::{Simd, SimdBase, f64x2};

/// Entropy source for the initial centroid draw.
///
/// The clustering algorithm below is shared, and only the unseeded random stream
/// differs per target. On WebAssembly `Random::new()` reads `Math.random`,
/// because that is what callers replace when not passing an explicit seed.
/// Natively `Random::new()` uses a fixed seeded stream so the test suite and
/// benchmarks stay repeatable. `Random::from_seed` uses the same SplitMix64
/// stream on every target.
#[cfg(target_arch = "wasm32")]
pub enum Random {
    Js,
    Seeded(u64),
}

#[cfg(target_arch = "wasm32")]
impl Random {
    pub fn new() -> Self {
        Random::Js
    }

    pub fn from_seed(seed: u32) -> Self {
        Random::Seeded(u64::from(seed) ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub fn next(&mut self) -> f64 {
        match self {
            Random::Js => js_sys::Math::random(),
            Random::Seeded(state) => splitmix64_next(state),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub enum Random {
    Default(rand::rngs::SmallRng),
    Seeded(u64),
}

#[cfg(not(target_arch = "wasm32"))]
impl Random {
    pub fn new() -> Self {
        use rand::SeedableRng;

        // Fixed seed so the native test suite and the benchmarks stay repeatable.
        Random::Default(rand::rngs::SmallRng::seed_from_u64(0))
    }

    pub fn from_seed(seed: u32) -> Self {
        Random::Seeded(u64::from(seed) ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub fn next(&mut self) -> f64 {
        match self {
            Random::Default(rng) => rand::RngExt::random(rng),
            Random::Seeded(state) => splitmix64_next(state),
        }
    }
}

fn splitmix64_next(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^= mixed >> 31;
    (mixed >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
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
    seed: Option<u32>,
) -> Vec<f64> {
    let mut random = match seed {
        Some(seed) => Random::from_seed(seed),
        None => Random::new(),
    };
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

    let point = &point[..dimensions];
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

/// Evaluates squared distances from `point` to 4 centroids in parallel with
/// independent `f64x2` accumulators while matching `get_distance_squared`
/// bit-for-bit.
#[inline]
pub fn distances_4x_f64<S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    dimensions: usize,
    offsets: [usize; 4],
) -> [f64; 4] {
    #[cfg(feature = "counters")]
    crate::work::count(&crate::work::DISTANCE_EVALUATIONS, 4);

    let point = &point[..dimensions];
    let c0 = &centroids[offsets[0]..offsets[0] + dimensions];
    let c1 = &centroids[offsets[1]..offsets[1] + dimensions];
    let c2 = &centroids[offsets[2]..offsets[2] + dimensions];
    let c3 = &centroids[offsets[3]..offsets[3] + dimensions];
    let even = dimensions - (dimensions % 2);
    let zero = f64x2::<S>::splat(simd, 0.0);
    let mut a0 = zero;
    let mut a1 = zero;
    let mut a2 = zero;
    let mut a3 = zero;

    let mut part = 0;
    while part < even {
        let p = f64x2::<S>::from_slice(simd, &point[part..part + 2]);
        let d0 = p - f64x2::<S>::from_slice(simd, &c0[part..part + 2]);
        let d1 = p - f64x2::<S>::from_slice(simd, &c1[part..part + 2]);
        let d2 = p - f64x2::<S>::from_slice(simd, &c2[part..part + 2]);
        let d3 = p - f64x2::<S>::from_slice(simd, &c3[part..part + 2]);
        a0 += d0 * d0;
        a1 += d1 * d1;
        a2 += d2 * d2;
        a3 += d3 * d3;
        part += 2;
    }

    let mut s0 = a0[0] + a0[1];
    let mut s1 = a1[0] + a1[1];
    let mut s2 = a2[0] + a2[1];
    let mut s3 = a3[0] + a3[1];

    if even < dimensions {
        let pv = point[even];
        let d0 = pv - c0[even];
        let d1 = pv - c1[even];
        let d2 = pv - c2[even];
        let d3 = pv - c3[even];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    [s0, s1, s2, s3]
}
