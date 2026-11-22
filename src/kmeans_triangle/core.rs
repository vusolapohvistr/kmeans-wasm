//! Hamerly k-means core: public entry points, result shape, and the main loop.

use fearless_simd::{Level, Simd, dispatch};

use super::initialize::{get_min_centroid_skip_point_centroid, initialize, point_all_centers};
use super::update::{Optional, move_centers, update_bounds};
use super::util::{get_centroids, get_distance_squared};

pub struct HamerlyKmeansResult {
    /// Flat row-major centroids, `k * dimensions` values.
    pub centroids: Vec<f64>,
    pub dimensions: usize,
    pub iterations: usize,
    point_states: Vec<super::initialize::PointState>,
}

impl HamerlyKmeansResult {
    /// One array per cluster, as the general `kmeans` API returns it.
    ///
    /// The packed color paths never need this shape, so it is built on demand
    /// rather than on every clustering run.
    pub fn centroid_rows(&self) -> Vec<Vec<f64>> {
        self.centroids
            .chunks_exact(self.dimensions)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// The cluster index of every point, as the general `kmeans` API returns it.
    ///
    /// Skipped by the packed color paths, which only want the palette.
    pub fn assignments(&self) -> Vec<u32> {
        self.point_states
            .iter()
            .map(|state| state.centroid)
            .collect()
    }
}

/// Runs the clustering once, at the strongest SIMD level this build supports.
///
/// The package is compiled with `simd128` required, so the level is a compile
/// time constant and there is nothing to detect at runtime. Selecting it here
/// means the level check happens once per clustering run rather than once per
/// distance evaluation.
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
///
/// A flat buffer keeps every point and every centroid in one contiguous
/// allocation, so the distance loops below stream memory instead of chasing a
/// pointer per point.
///
/// The distance kernel is generic over the SIMD level, so the caller selects
/// the level once with `dispatch!` instead of paying for a level check on every
/// one of the millions of distance evaluations.
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
        None => get_centroids(points, dimensions, point_count, k, weights),
    };
    let k = k.min(centroids.len() / dimensions.max(1)).max(1);

    let super::initialize::InitializeResult {
        mut centroid_points_counts,
        mut centroid_points_sum,
        mut point_states,
    } = initialize(simd, &centroids, points, dimensions, point_count, weights);

    let mut centroid_closest_centroid_distance = vec![f64::MAX; k];
    let mut centroid_distance_to_previous_position = vec![f64::MAX; k];
    let mut iterations = 0;

    while iterations < max_iter {
        for (j, slot) in centroid_closest_centroid_distance.iter_mut().enumerate() {
            *slot = get_min_centroid_skip_point_centroid(
                simd,
                &centroids[j * dimensions..(j + 1) * dimensions],
                &centroids,
                dimensions,
                k,
                j,
            );
        }

        for (i, state) in point_states.iter_mut().enumerate() {
            #[cfg(feature = "counters")]
            crate::work::count_one(&crate::work::BOUND_CHECKS);
            let point_offset = i * dimensions;

            let m = f64::max(
                centroid_closest_centroid_distance[state.centroid as usize] / 2.0,
                state.lower_bound,
            );
            if state.upper_bound > m {
                let current = state.centroid as usize;
                let current_distance_squared = get_distance_squared(
                    simd,
                    &points[point_offset..],
                    &centroids,
                    dimensions,
                    current * dimensions,
                );
                state.upper_bound = current_distance_squared.sqrt();
                if state.upper_bound > m {
                    #[cfg(feature = "counters")]
                    crate::work::count_one(&crate::work::FULL_SCANS);
                    let previous_point_centroid = state.centroid;
                    point_all_centers(
                        simd,
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
                        let weight = weights.map_or(1.0, |all| all[i]);
                        #[cfg(feature = "counters")]
                        crate::work::count(&crate::work::SUM_UPDATES, 2 * dimensions as u64 + 2);
                        centroid_points_counts[previous_point_centroid as usize] -= weight;
                        for part in 0..dimensions {
                            let value = weight * points[point_offset + part];
                            centroid_points_sum[previous + part] -= value;
                            centroid_points_sum[current + part] += value;
                        }
                        centroid_points_counts[state.centroid as usize] += weight;
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
        centroids,
        dimensions,
        iterations,
        point_states,
    }
}
