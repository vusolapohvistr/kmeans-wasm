pub struct HamerlyKmeansResult {
    pub centroids: Vec<Vec<f64>>,
    pub iterations: usize,
    pub point_centroids: Vec<u32>,
}

/// Cluster `points`, stored as a flat row-major buffer of `dimensions` components
/// per point.
///
/// A flat buffer keeps every point and every centroid in one contiguous
/// allocation, so the distance loops below stream memory instead of chasing a
/// pointer per point.
pub fn hamerly_kmeans(
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
    points: &[f64],
    dimensions: usize,
) -> HamerlyKmeansResult {
    let point_count = points.len().checked_div(dimensions).unwrap_or(0);

    if point_count == 0 {
        return HamerlyKmeansResult {
            centroids: Vec::new(),
            iterations: 0,
            point_centroids: Vec::new(),
        };
    }

    let mut centroids = get_centroids(points, dimensions, point_count, k);

    let InitializeResult {
        mut centroid_points_counts,
        mut centroid_points_sum,
        mut point_states,
    } = initialize(&centroids, points, dimensions, point_count);

    let mut centroid_closest_centroid_distance = vec![f64::MAX; k];
    let mut centroid_distance_to_previous_position = vec![f64::MAX; k];
    let mut iterations = 0;

    while iterations < max_iter {
        for (j, slot) in centroid_closest_centroid_distance.iter_mut().enumerate() {
            *slot = get_min_centroid_skip_point_centroid(
                &centroids[j * dimensions..(j + 1) * dimensions],
                &centroids,
                dimensions,
                k,
                j,
            );
        }

        for (i, state) in point_states.iter_mut().enumerate() {
            let point_offset = i * dimensions;

            let m = f64::max(
                centroid_closest_centroid_distance[state.centroid as usize] / 2.0,
                state.lower_bound,
            );
            if state.upper_bound > m {
                let current = state.centroid as usize;
                let current_distance_squared = get_distance_squared(
                    &points[point_offset..],
                    &centroids,
                    dimensions,
                    current * dimensions,
                );
                state.upper_bound = current_distance_squared.sqrt();
                if state.upper_bound > m {
                    let previous_point_centroid = state.centroid;
                    point_all_centers(
                        &points[point_offset..],
                        &centroids,
                        dimensions,
                        k,
                        Some((current, current_distance_squared)),
                        state,
                    );
                    if previous_point_centroid != state.centroid {
                        let previous = previous_point_centroid as usize * dimensions;
                        let current = state.centroid as usize * dimensions;
                        centroid_points_counts[previous_point_centroid as usize] -= 1;
                        for part in 0..dimensions {
                            let value = points[point_offset + part];
                            centroid_points_sum[previous + part] -= value;
                            centroid_points_sum[current + part] += value;
                        }
                        centroid_points_counts[state.centroid as usize] += 1;
                    }
                }
            }
        }

        let total_squared_distance_moved = move_centers(
            &mut centroids,
            &centroid_points_sum,
            &centroid_points_counts,
            dimensions,
            k,
            &mut centroid_distance_to_previous_position,
        );
        update_bounds(&centroid_distance_to_previous_position, &mut point_states);

        if total_squared_distance_moved < convergence_threshold {
            break;
        }

        iterations += 1;
    }

    HamerlyKmeansResult {
        centroids: centroids
            .chunks_exact(dimensions)
            .map(<[f64]>::to_vec)
            .collect(),
        iterations,
        point_centroids: point_states.iter().map(|state| state.centroid).collect(),
    }
}

fn update_bounds(
    centroid_distance_to_previous_position: &[f64], // p(j) – distance that c(j) last moved
    point_states: &mut [PointState],
) {
    let r = centroid_distance_to_previous_position
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let r_another = centroid_distance_to_previous_position
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != r)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;

    let r_moved = centroid_distance_to_previous_position[r];
    let r_another_moved = centroid_distance_to_previous_position[r_another];

    // The two cases used to be a branch per point. Splitting the pass keeps the
    // inner loop straight-line, and the arithmetic applied to each point is
    // unchanged.
    for state in point_states.iter_mut() {
        state.upper_bound += centroid_distance_to_previous_position[state.centroid as usize];
    }
    for state in point_states.iter_mut() {
        state.lower_bound -= if state.centroid as usize == r {
            r_another_moved
        } else {
            r_moved
        };
    }
}

fn move_centers(
    centroids: &mut [f64],
    centroid_points_sum: &[f64],
    centroid_points_counts: &[usize],
    dimensions: usize,
    k: usize,
    centroid_distance_to_previous_position: &mut [f64],
) -> f64 {
    let mut total_squared_distance_moved = 0.0;
    for j in 0..k {
        let centroid_offset = j * dimensions;
        let points_count = centroid_points_counts[j] as f64;
        let mut squared_distance_moved = 0.0;
        for part in 0..dimensions {
            let index = centroid_offset + part;
            let previous_position = centroids[index];
            let moved = centroid_points_sum[index] / points_count;
            centroids[index] = moved;
            let difference = previous_position - moved;
            squared_distance_moved += difference * difference;
        }
        centroid_distance_to_previous_position[j] = squared_distance_moved.sqrt();
        total_squared_distance_moved += squared_distance_moved;
    }
    total_squared_distance_moved
}

/// Per-point state, kept in one allocation instead of three parallel arrays.
///
/// The main loop and the bounds update touch the assignment, the upper bound and
/// the lower bound of the same point together, so interleaving them keeps one
/// point inside a single cache line instead of pulling in three far-apart
/// addresses.
#[derive(Clone, Copy)]
struct PointState {
    lower_bound: f64, // l(i) lower bound on the distance between x(i) and its second closest center – that is, the closest center to x(i) that is not c(a(i))
    upper_bound: f64, // u(i) upper bound on the distance between x(i) and its assigned center c(a(i)),
    centroid: u32,    // a(i) index of the center to which x(i) is assigned
}

struct InitializeResult {
    centroid_points_counts: Vec<usize>, // q(j) – number of points assigned to cluster j
    centroid_points_sum: Vec<f64>,      // c`(j) vector sum of all points in cluster j
    point_states: Vec<PointState>,
}
fn initialize(
    centroids: &[f64],
    points: &[f64],
    dimensions: usize,
    point_count: usize,
) -> InitializeResult {
    let k = centroids.len() / dimensions;

    let mut centroid_points_counts = vec![0; k];
    let mut centroid_points_sum = vec![0.0; k * dimensions];
    let mut point_states = vec![
        PointState {
            lower_bound: 0.0,
            upper_bound: 0.0,
            centroid: 0,
        };
        point_count
    ];

    for (i, state) in point_states.iter_mut().enumerate() {
        let point_offset = i * dimensions;
        point_all_centers(
            &points[point_offset..],
            centroids,
            dimensions,
            k,
            None,
            state,
        );
        let sum_offset = state.centroid as usize * dimensions;
        centroid_points_counts[state.centroid as usize] += 1;
        for part in 0..dimensions {
            centroid_points_sum[sum_offset + part] += points[point_offset + part];
        }
    }

    InitializeResult {
        centroid_points_counts,
        centroid_points_sum,
        point_states,
    }
}

/// Finds the closest and the second closest centroid for one point in a single
/// pass over the centroids.
///
/// The second pass used to run over the centroids a second time, which doubled
/// the distance work in the initialization and in every reassignment. Tracking
/// the runner-up while scanning gives the same two distances: the closest
/// centroid is still the first index holding the minimum, and the runner-up
/// distance is the smallest distance among the remaining centroids.
///
/// `known_centroid` lets the caller hand over a distance it already measured for
/// one of the centroids, instead of having it computed twice. The scan order is
/// unchanged, so ties still resolve to the lowest index.
#[inline]
fn point_all_centers(
    point: &[f64],
    centroids: &[f64],
    dimensions: usize,
    k: usize,
    known_centroid: Option<(usize, f64)>,
    state: &mut PointState,
) {
    let mut min_index = 0;
    let mut min_distance_squared = f64::MAX;
    let mut second_distance_squared = f64::MAX;

    for j in 0..k {
        let distance_squared = match known_centroid {
            Some((known, distance)) if known == j => distance,
            _ => get_distance_squared(point, centroids, dimensions, j * dimensions),
        };

        if distance_squared < min_distance_squared {
            second_distance_squared = min_distance_squared;
            min_distance_squared = distance_squared;
            min_index = j;
        } else if distance_squared < second_distance_squared {
            second_distance_squared = distance_squared;
        }
    }

    state.centroid = min_index as u32;
    state.upper_bound = min_distance_squared.sqrt();
    state.lower_bound = second_distance_squared.sqrt();
}

fn get_min_centroid_skip_point_centroid(
    point: &[f64],
    centroids: &[f64],
    dimensions: usize,
    k: usize,
    point_centroid: usize,
) -> f64 {
    let mut min_distance_squared = f64::MAX;

    for j in 0..k {
        if j == point_centroid {
            continue;
        }

        let distance_squared = get_distance_squared(point, centroids, dimensions, j * dimensions);
        if distance_squared < min_distance_squared {
            min_distance_squared = distance_squared;
        }
    }

    min_distance_squared.sqrt()
}

#[inline]
fn get_distance_squared(
    point: &[f64],
    centroids: &[f64],
    dimensions: usize,
    centroid_offset: usize,
) -> f64 {
    // Indexed rather than iterator based: an interleaved benchmark of the
    // alternatives (zip/map/sum, zip with an explicit loop, chunks_exact and
    // plain indexing) put them within noise for 3 and 8 components, and plain
    // indexing ahead by about 4% at 50 components. It is also the shortest form.
    let centroid = &centroids[centroid_offset..centroid_offset + dimensions];
    let mut sum = 0.0;
    for part in 0..dimensions {
        let difference = point[part] - centroid[part];
        sum += difference * difference;
    }
    sum
}

/// Entropy source for the initial centroid draw.
///
/// The clustering algorithm below is shared, and only the random stream differs
/// per target. On WebAssembly it has to come from `Math.random`, because that is
/// what callers replace to make a run reproducible, for example the comparison
/// page seeds it before every measurement. Natively it is a seeded stream so the
/// test suite and the benchmarks stay repeatable.
#[cfg(target_arch = "wasm32")]
struct Random;

#[cfg(target_arch = "wasm32")]
impl Random {
    fn new() -> Self {
        Random
    }

    fn next(&mut self) -> f64 {
        js_sys::Math::random()
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct Random(rand::rngs::SmallRng);

#[cfg(not(target_arch = "wasm32"))]
impl Random {
    fn new() -> Self {
        use rand::SeedableRng;

        // Fixed seed so the native test suite and the benchmarks stay repeatable.
        Random(rand::rngs::SmallRng::seed_from_u64(0))
    }

    fn next(&mut self) -> f64 {
        rand::RngExt::random(&mut self.0)
    }
}

/// Picks the initial centroids as `k` distinct input points.
fn get_centroids(points: &[f64], dimensions: usize, point_count: usize, k: usize) -> Vec<f64> {
    let distinct = k.min(point_count);
    let mut random = Random::new();
    let mut chosen: Vec<usize> = Vec::with_capacity(distinct);

    // Rejection sampling, the same draw on every target. When k is above the
    // number of input points there is nothing left to draw, so the remainder
    // repeats the last pick rather than rejecting forever.
    while chosen.len() < distinct {
        let candidate = (random.next() * (point_count as f64)) as usize;
        if !chosen.contains(&candidate) {
            chosen.push(candidate);
        }
    }
    chosen.resize(k, *chosen.last().expect("point_count is not zero"));

    let mut centroids = Vec::with_capacity(k * dimensions);
    for index in chosen {
        let start = index * dimensions;
        centroids.extend_from_slice(&points[start..start + dimensions]);
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::hamerly_kmeans;

    fn points(count: usize, dimensions: usize) -> Vec<f64> {
        let mut state = 0x12345678_u64;
        (0..count * dimensions)
            .map(|_| {
                state = (state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407))
                    >> 33;
                state as f64 / 512.0
            })
            .collect()
    }

    // FNV-1a over the exact bit patterns of the iteration count, the centroids
    // and every assignment, so any change in the clustering arithmetic shows up.
    fn fingerprint(count: usize, dimensions: usize, k: usize) -> String {
        let result = hamerly_kmeans(k, 100, 0.1, &points(count, dimensions), dimensions);
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;

        let mut eat = |value: u64| {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };

        eat(result.iterations as u64);
        eat(result.centroids.len() as u64);
        for centroid in &result.centroids {
            for value in centroid {
                eat(value.to_bits());
            }
        }
        for index in &result.point_centroids {
            eat(u64::from(*index));
        }

        format!("{hash:016x}")
    }

    #[test]
    fn keeps_the_recorded_clustering_output() {
        // Recorded with the shared centroid initialization, which is the one the
        // WebAssembly build uses. The flat buffer and the single-pass centroid
        // scan are optimizations, so they have to reproduce these values bit for
        // bit. Re-record them only when a change to the arithmetic is intended.
        let cases = [
            (2_000, 3, 4, "3f7a25b335403c80"),
            (2_000, 3, 16, "ccdbd311ee118ba0"),
            (5_000, 8, 12, "b8a86a71b62cdcbc"),
            (1_000, 16, 32, "ac36c7981cf56b61"),
            (200, 50, 8, "e2ef3bdb97c3189a"),
        ];

        for (count, dimensions, clusters, expected) in cases {
            assert_eq!(
                fingerprint(count, dimensions, clusters),
                expected,
                "{count} points, {dimensions} dimensions, k={clusters}"
            );
        }
    }

    #[test]
    fn keeps_the_point_dimension_for_every_centroid() {
        let result = hamerly_kmeans(6, 50, 0.1, &points(500, 7), 7);

        assert_eq!(result.centroids.len(), 6);
        for centroid in &result.centroids {
            assert_eq!(centroid.len(), 7);
        }
    }

    #[test]
    fn returns_nothing_for_empty_input() {
        let result = hamerly_kmeans(3, 10, 0.1, &[], 3);

        assert!(result.centroids.is_empty());
        assert!(result.point_centroids.is_empty());
        assert_eq!(result.iterations, 0);
    }
}
