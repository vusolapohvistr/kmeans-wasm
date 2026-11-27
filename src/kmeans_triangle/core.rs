//! Hamerly k-means core: public entry points, result shape, and the main loop.

use fearless_simd::{Level, Simd, dispatch};

use super::initialize::{
    Neighbor, PointState, distance_squared_fixed, initialize, point_all_centers_fixed,
    point_all_centers_generic, point_all_centers_pruned_fixed, point_all_centers_pruned_generic,
    transpose_centroids,
};
use super::update::{Optional, move_centers, top_two_moves};
use super::util::{get_centroids, get_distance_squared};

pub struct HamerlyKmeansResult {
    /// Flat row-major centroids, `k * dimensions` values.
    pub centroids: Vec<f64>,
    pub dimensions: usize,
    pub iterations: usize,
    point_states: Vec<PointState>,
}

impl HamerlyKmeansResult {
    /// One array per cluster, as the general `kmeans` API returns it.
    pub fn centroid_rows(&self) -> Vec<Vec<f64>> {
        self.centroids
            .chunks_exact(self.dimensions)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// The cluster index of every point, as the general `kmeans` API returns it.
    pub fn assignments(&self) -> Vec<u32> {
        self.point_states
            .iter()
            .map(|state| state.centroid)
            .collect()
    }
}

/// Runs the clustering once, at the strongest SIMD level this build supports.
pub fn hamerly_kmeans_dispatched(
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
    points: &[f64],
    dimensions: usize,
) -> HamerlyKmeansResult {
    hamerly_kmeans_dispatched_weighted(
        k,
        max_iter,
        convergence_threshold,
        points,
        dimensions,
        Optional::default(),
    )
}

pub fn hamerly_kmeans_dispatched_weighted(
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
    points: &[f64],
    dimensions: usize,
    optional: Optional<'_>,
) -> HamerlyKmeansResult {
    let level = Level::baseline();
    dispatch!(level, simd => hamerly_kmeans_weighted(
        simd,
        k,
        max_iter,
        convergence_threshold,
        points,
        dimensions,
        optional
    ))
}

/// Cluster `points`, stored as a flat row-major buffer of `dimensions` components
/// per point.
pub fn hamerly_kmeans<S: Simd>(
    simd: S,
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
    points: &[f64],
    dimensions: usize,
) -> HamerlyKmeansResult {
    hamerly_kmeans_weighted(
        simd,
        k,
        max_iter,
        convergence_threshold,
        points,
        dimensions,
        Optional::default(),
    )
}

/// As `hamerly_kmeans`, but with the optional inputs supplied.
pub fn hamerly_kmeans_weighted<S: Simd>(
    simd: S,
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
    points: &[f64],
    dimensions: usize,
    optional: Optional<'_>,
) -> HamerlyKmeansResult {
    let Optional {
        weights,
        initial_centroids,
        seed,
    } = optional;
    let point_count = points.len().checked_div(dimensions).unwrap_or(0);

    if point_count == 0 {
        return HamerlyKmeansResult {
            centroids: Vec::new(),
            dimensions,
            iterations: 0,
            point_states: Vec::new(),
        };
    }

    let mut centroids = match initial_centroids {
        Some(given) => given.to_vec(),
        None => get_centroids(points, dimensions, point_count, k, weights, seed),
    };
    let k = k.min(centroids.len() / dimensions.max(1)).max(1);

    let keep_transposed = matches!(dimensions, 3 | 4) && k < 8;
    let mut transposed = if keep_transposed {
        let mut buffer = vec![0.0; dimensions * k];
        transpose_centroids(&centroids, dimensions, k, &mut buffer);
        buffer
    } else {
        Vec::new()
    };

    let stride = k.saturating_sub(1);
    let mut sorted_neighbors = vec![
        Neighbor {
            dist: 0.0,
            index: 0
        };
        k * stride
    ];
    // `(move_c, half_min_c)` packed per centroid so `assign_round_*` loads both in one read.
    let mut centroid_info = vec![(0.0, f64::MAX); k];
    if k >= 2 {
        compute_centroid_matrix(
            simd,
            &centroids,
            dimensions,
            k,
            &mut sorted_neighbors,
            &mut centroid_info,
        );
    }

    let super::initialize::InitializeResult {
        mut centroid_points_counts,
        mut centroid_points_sum,
        mut point_states,
    } = initialize(
        simd,
        &centroids,
        &transposed,
        &sorted_neighbors,
        points,
        dimensions,
        point_count,
        weights,
    );

    let mut centroid_distance_to_previous_position = vec![0.0; k];
    let mut pending_top_two = (0u32, 0.0f64, 0.0f64);
    let mut iterations = 0;

    while iterations < max_iter {
        if iterations > 0 && k >= 2 {
            compute_centroid_matrix(
                simd,
                &centroids,
                dimensions,
                k,
                &mut sorted_neighbors,
                &mut centroid_info,
            );
        }

        let (r_u32, move_r, move_r2) = pending_top_two;

        match dimensions {
            3 => assign_round_fixed::<3, _>(
                simd,
                points,
                &centroids,
                &transposed,
                &sorted_neighbors,
                k,
                weights,
                &centroid_info,
                iterations > 0,
                r_u32,
                move_r,
                move_r2,
                &mut point_states,
                &mut centroid_points_counts,
                &mut centroid_points_sum,
            ),
            4 => assign_round_fixed::<4, _>(
                simd,
                points,
                &centroids,
                &transposed,
                &sorted_neighbors,
                k,
                weights,
                &centroid_info,
                iterations > 0,
                r_u32,
                move_r,
                move_r2,
                &mut point_states,
                &mut centroid_points_counts,
                &mut centroid_points_sum,
            ),
            _ => assign_round_generic(
                simd,
                points,
                &centroids,
                &sorted_neighbors,
                dimensions,
                k,
                weights,
                &centroid_info,
                iterations > 0,
                r_u32,
                move_r,
                move_r2,
                &mut point_states,
                &mut centroid_points_counts,
                &mut centroid_points_sum,
            ),
        }

        let total_squared_distance_moved = move_centers(
            &mut centroids,
            &centroid_points_sum,
            &centroid_points_counts,
            dimensions,
            k,
            &mut centroid_distance_to_previous_position,
        );

        if total_squared_distance_moved < convergence_threshold {
            break;
        }

        pending_top_two =
            top_two_moves(&centroid_distance_to_previous_position).unwrap_or((0, 0.0, 0.0));
        for (info, &moved) in centroid_info
            .iter_mut()
            .zip(centroid_distance_to_previous_position.iter())
        {
            info.0 = moved;
        }
        if keep_transposed {
            transpose_centroids(&centroids, dimensions, k, &mut transposed);
        }
        iterations += 1;
    }

    HamerlyKmeansResult {
        centroids,
        dimensions,
        iterations,
        point_states,
    }
}

/// Computes all pairwise inter-centroid distances, sorts each row ascending in
/// `sorted_neighbors` (`k * (k - 1)`), and records `s(i) / 2` in `centroid_info[i].1`.
fn compute_centroid_matrix<S: Simd>(
    simd: S,
    centroids: &[f64],
    dimensions: usize,
    k: usize,
    sorted_neighbors: &mut [Neighbor],
    centroid_info: &mut [(f64, f64)],
) {
    let stride = k - 1;
    for i in 0..k {
        let ci = &centroids[i * dimensions..(i + 1) * dimensions];
        for j in (i + 1)..k {
            let d = get_distance_squared(simd, ci, centroids, dimensions, j * dimensions).sqrt();
            sorted_neighbors[i * stride + (j - 1)] = Neighbor {
                dist: d,
                index: j as u32,
            };
            sorted_neighbors[j * stride + i] = Neighbor {
                dist: d,
                index: i as u32,
            };
        }
        let row = &mut sorted_neighbors[i * stride..(i + 1) * stride];
        row.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        centroid_info[i].1 = 0.5 * row[0].dist;
    }
}

#[allow(clippy::too_many_arguments)]
fn assign_round_fixed<const D: usize, S: Simd>(
    simd: S,
    points: &[f64],
    centroids: &[f64],
    transposed: &[f64],
    sorted_neighbors: &[Neighbor],
    k: usize,
    weights: Option<&[f64]>,
    centroid_info: &[(f64, f64)],
    count_bound_updates: bool,
    r_u32: u32,
    move_r: f64,
    move_r2: f64,
    point_states: &mut [PointState],
    centroid_points_counts: &mut [f64],
    centroid_points_sum: &mut [f64],
) {
    #[cfg(feature = "counters")]
    if count_bound_updates {
        crate::work::count(&crate::work::BOUND_UPDATES, point_states.len() as u64);
    }
    #[cfg(not(feature = "counters"))]
    let _ = count_bound_updates;

    let stride = k.saturating_sub(1);
    for (i, state) in point_states.iter_mut().enumerate() {
        let (move_c, half_min) = centroid_info[state.centroid as usize];
        state.upper_bound += move_c;
        let subtract = if state.centroid == r_u32 {
            move_r2
        } else {
            move_r
        };
        state.lower_bound -= subtract;

        #[cfg(feature = "counters")]
        crate::work::count_one(&crate::work::BOUND_CHECKS);

        let m = f64::max(half_min, state.lower_bound);
        if state.upper_bound > m {
            let point_offset = i * D;
            let point = &points[point_offset..point_offset + D];
            let current = state.centroid as usize;
            let current_offset = current * D;
            let current_distance_squared = distance_squared_fixed::<D, _>(
                simd,
                point,
                &centroids[current_offset..current_offset + D],
            );
            let u = current_distance_squared.sqrt();
            state.upper_bound = u;
            if u > m {
                #[cfg(feature = "counters")]
                crate::work::count_one(&crate::work::FULL_SCANS);
                let previous_point_centroid = state.centroid;
                if k >= 8 {
                    point_all_centers_pruned_fixed::<D, _>(
                        simd,
                        point,
                        centroids,
                        &sorted_neighbors[current * stride..(current + 1) * stride],
                        current,
                        current_distance_squared,
                        u,
                        state,
                    );
                } else {
                    point_all_centers_fixed::<D, _>(
                        simd,
                        point,
                        centroids,
                        transposed,
                        k,
                        Some((current, current_distance_squared)),
                        state,
                    );
                }
                if previous_point_centroid != state.centroid {
                    let previous = previous_point_centroid as usize * D;
                    let current = state.centroid as usize * D;
                    let weight = weights.map_or(1.0, |all| all[i]);
                    #[cfg(feature = "counters")]
                    crate::work::count(&crate::work::SUM_UPDATES, 2 * D as u64 + 2);
                    centroid_points_counts[previous_point_centroid as usize] -= weight;
                    for part in 0..D {
                        let value = weight * point[part];
                        centroid_points_sum[previous + part] -= value;
                        centroid_points_sum[current + part] += value;
                    }
                    centroid_points_counts[state.centroid as usize] += weight;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn assign_round_generic<S: Simd>(
    simd: S,
    points: &[f64],
    centroids: &[f64],
    sorted_neighbors: &[Neighbor],
    dimensions: usize,
    k: usize,
    weights: Option<&[f64]>,
    centroid_info: &[(f64, f64)],
    count_bound_updates: bool,
    r_u32: u32,
    move_r: f64,
    move_r2: f64,
    point_states: &mut [PointState],
    centroid_points_counts: &mut [f64],
    centroid_points_sum: &mut [f64],
) {
    #[cfg(feature = "counters")]
    if count_bound_updates {
        crate::work::count(&crate::work::BOUND_UPDATES, point_states.len() as u64);
    }
    #[cfg(not(feature = "counters"))]
    let _ = count_bound_updates;

    let stride = k.saturating_sub(1);
    for (i, state) in point_states.iter_mut().enumerate() {
        let (move_c, half_min) = centroid_info[state.centroid as usize];
        state.upper_bound += move_c;
        let subtract = if state.centroid == r_u32 {
            move_r2
        } else {
            move_r
        };
        state.lower_bound -= subtract;

        #[cfg(feature = "counters")]
        crate::work::count_one(&crate::work::BOUND_CHECKS);

        let m = f64::max(half_min, state.lower_bound);
        if state.upper_bound > m {
            let point_offset = i * dimensions;
            let point = &points[point_offset..point_offset + dimensions];
            let current = state.centroid as usize;
            let current_distance_squared =
                get_distance_squared(simd, point, centroids, dimensions, current * dimensions);
            let u = current_distance_squared.sqrt();
            state.upper_bound = u;
            if u > m {
                #[cfg(feature = "counters")]
                crate::work::count_one(&crate::work::FULL_SCANS);
                let previous_point_centroid = state.centroid;
                if k >= 4 {
                    point_all_centers_pruned_generic(
                        simd,
                        point,
                        centroids,
                        dimensions,
                        &sorted_neighbors[current * stride..(current + 1) * stride],
                        current,
                        current_distance_squared,
                        u,
                        state,
                    );
                } else {
                    point_all_centers_generic(
                        simd,
                        point,
                        centroids,
                        dimensions,
                        k,
                        Some((current, current_distance_squared)),
                        state,
                    );
                }
                if previous_point_centroid != state.centroid {
                    let previous = previous_point_centroid as usize * dimensions;
                    let current = state.centroid as usize * dimensions;
                    let weight = weights.map_or(1.0, |all| all[i]);
                    #[cfg(feature = "counters")]
                    crate::work::count(&crate::work::SUM_UPDATES, 2 * dimensions as u64 + 2);
                    centroid_points_counts[previous_point_centroid as usize] -= weight;
                    for part in 0..dimensions {
                        let value = weight * point[part];
                        centroid_points_sum[previous + part] -= value;
                        centroid_points_sum[current + part] += value;
                    }
                    centroid_points_counts[state.centroid as usize] += weight;
                }
            }
        }
    }
}
