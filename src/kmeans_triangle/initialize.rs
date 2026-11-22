//! Initialization phase: nearest and second-nearest centroid for every point,
//! plus the per-point state and sums used by the Hamerly loop.

use fearless_simd::Simd;

use super::util::get_distance_squared;

/// Per-point state, kept in one allocation instead of three parallel arrays.
///
/// The main loop and the bounds update touch the assignment, the upper bound and
/// the lower bound of the same point together, so interleaving them keeps one
/// point inside a single cache line instead of pulling in three far-apart
/// addresses.
#[derive(Clone, Copy)]
pub(crate) struct PointState {
    pub(crate) lower_bound: f64, // l(i) lower bound on the distance between x(i) and its second closest center – that is, the closest center to x(i) that is not c(a(i))
    pub(crate) upper_bound: f64, // u(i) upper bound on the distance between x(i) and its assigned center c(a(i)),
    pub(crate) centroid: u32,    // a(i) index of the center to which x(i) is assigned
}

pub(super) struct InitializeResult {
    pub(super) centroid_points_counts: Vec<f64>, // q(j) – total weight assigned to cluster j
    pub(super) centroid_points_sum: Vec<f64>,    // c`(j) vector sum of all points in cluster j
    pub(super) point_states: Vec<PointState>,
}

pub(super) fn initialize<S: Simd>(
    simd: S,
    centroids: &[f64],
    points: &[f64],
    dimensions: usize,
    point_count: usize,
    weights: Option<&[f64]>,
) -> InitializeResult {
    let k = centroids.len() / dimensions;

    let mut centroid_points_counts = vec![0.0; k];
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
            simd,
            &points[point_offset..],
            centroids,
            dimensions,
            k,
            None,
            state,
        );
        let sum_offset = state.centroid as usize * dimensions;
        let weight = weights.map_or(1.0, |all| all[i]);
        centroid_points_counts[state.centroid as usize] += weight;
        for part in 0..dimensions {
            centroid_points_sum[sum_offset + part] += weight * points[point_offset + part];
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
pub(super) fn point_all_centers<S: Simd>(
    simd: S,
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
            _ => get_distance_squared(simd, point, centroids, dimensions, j * dimensions),
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

pub(super) fn get_min_centroid_skip_point_centroid<S: Simd>(
    simd: S,
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

        let distance_squared =
            get_distance_squared(simd, point, centroids, dimensions, j * dimensions);
        if distance_squared < min_distance_squared {
            min_distance_squared = distance_squared;
        }
    }

    min_distance_squared.sqrt()
}
