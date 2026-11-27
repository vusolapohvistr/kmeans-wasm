//! Initialization phase: nearest and second-nearest centroid for every point,
//! plus the per-point state and sums used by the Hamerly loop.

use fearless_simd::{Simd, SimdBase, f64x2};

use super::util::{distances_4x_f64, get_distance_squared};

/// Pre-sorted inter-centroid neighbor entry used for early-exit triangle pruning.
#[derive(Clone, Copy)]
pub(super) struct Neighbor {
    pub(super) dist: f64,
    pub(super) index: u32,
}

/// Per-point state, kept in one allocation instead of three parallel arrays.
///
/// The main loop and the bounds update touch the assignment, the upper bound and
/// the lower bound of the same point together, so interleaving them keeps one
/// point inside a single cache line instead of pulling in three far-apart
/// addresses.
#[derive(Clone, Copy)]
pub(crate) struct PointState {
    pub(crate) lower_bound: f64, // l(i) lower bound on the distance between x(i) and its second closest center
    pub(crate) upper_bound: f64, // u(i) upper bound on the distance between x(i) and its assigned center
    pub(crate) centroid: u32,    // a(i) index of the center to which x(i) is assigned
}

pub(super) struct InitializeResult {
    pub(super) centroid_points_counts: Vec<f64>, // q(j) – total weight assigned to cluster j
    pub(super) centroid_points_sum: Vec<f64>,    // c`(j) vector sum of all points in cluster j
    pub(super) point_states: Vec<PointState>,
}

/// Transposes row-major centroids (`k * dimensions`) into dimension-major
/// layout (`dimensions * k`) so two consecutive clusters are contiguous per
/// dimension for `f64x2` SIMD scans.
pub(super) fn transpose_centroids(
    centroids: &[f64],
    dimensions: usize,
    k: usize,
    transposed: &mut [f64],
) {
    for (cluster, row) in centroids.chunks_exact(dimensions).enumerate() {
        for (part, value) in row.iter().enumerate() {
            transposed[part * k + cluster] = *value;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn initialize<S: Simd>(
    simd: S,
    centroids: &[f64],
    transposed: &[f64],
    sorted_neighbors: &[Neighbor],
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

    match dimensions {
        3 => initialize_fixed::<3, _>(
            simd,
            centroids,
            transposed,
            sorted_neighbors,
            points,
            k,
            weights,
            &mut centroid_points_counts,
            &mut centroid_points_sum,
            &mut point_states,
        ),
        4 => initialize_fixed::<4, _>(
            simd,
            centroids,
            transposed,
            sorted_neighbors,
            points,
            k,
            weights,
            &mut centroid_points_counts,
            &mut centroid_points_sum,
            &mut point_states,
        ),
        _ => initialize_generic(
            simd,
            centroids,
            sorted_neighbors,
            points,
            dimensions,
            k,
            weights,
            &mut centroid_points_counts,
            &mut centroid_points_sum,
            &mut point_states,
        ),
    }

    InitializeResult {
        centroid_points_counts,
        centroid_points_sum,
        point_states,
    }
}

#[allow(clippy::too_many_arguments)]
fn initialize_fixed<const D: usize, S: Simd>(
    simd: S,
    centroids: &[f64],
    transposed: &[f64],
    sorted_neighbors: &[Neighbor],
    points: &[f64],
    k: usize,
    weights: Option<&[f64]>,
    centroid_points_counts: &mut [f64],
    centroid_points_sum: &mut [f64],
    point_states: &mut [PointState],
) {
    let mut prev = 0usize;
    let stride = k.saturating_sub(1);
    for (i, state) in point_states.iter_mut().enumerate() {
        let point_offset = i * D;
        let point = &points[point_offset..point_offset + D];
        if k >= 8 {
            let prev_offset = prev * D;
            let d2 = distance_squared_fixed::<D, _>(
                simd,
                point,
                &centroids[prev_offset..prev_offset + D],
            );
            let u = d2.sqrt();
            point_all_centers_pruned_fixed::<D, _>(
                simd,
                point,
                centroids,
                &sorted_neighbors[prev * stride..(prev + 1) * stride],
                prev,
                d2,
                u,
                state,
            );
            prev = state.centroid as usize;
        } else {
            point_all_centers_fixed::<D, _>(simd, point, centroids, transposed, k, None, state);
        }
        let sum_offset = state.centroid as usize * D;
        let weight = weights.map_or(1.0, |all| all[i]);
        centroid_points_counts[state.centroid as usize] += weight;
        for part in 0..D {
            centroid_points_sum[sum_offset + part] += weight * point[part];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn initialize_generic<S: Simd>(
    simd: S,
    centroids: &[f64],
    sorted_neighbors: &[Neighbor],
    points: &[f64],
    dimensions: usize,
    k: usize,
    weights: Option<&[f64]>,
    centroid_points_counts: &mut [f64],
    centroid_points_sum: &mut [f64],
    point_states: &mut [PointState],
) {
    let mut prev = 0usize;
    let stride = k.saturating_sub(1);
    for (i, state) in point_states.iter_mut().enumerate() {
        let point_offset = i * dimensions;
        let point = &points[point_offset..point_offset + dimensions];
        if k >= 4 {
            let d2 = get_distance_squared(simd, point, centroids, dimensions, prev * dimensions);
            let u = d2.sqrt();
            point_all_centers_pruned_generic(
                simd,
                point,
                centroids,
                dimensions,
                &sorted_neighbors[prev * stride..(prev + 1) * stride],
                prev,
                d2,
                u,
                state,
            );
            prev = state.centroid as usize;
        } else {
            point_all_centers_generic(simd, point, centroids, dimensions, k, None, state);
        }
        let sum_offset = state.centroid as usize * dimensions;
        let weight = weights.map_or(1.0, |all| all[i]);
        centroid_points_counts[state.centroid as usize] += weight;
        for part in 0..dimensions {
            centroid_points_sum[sum_offset + part] += weight * point[part];
        }
    }
}

/// The running minimum and runner-up distances for one point.
#[derive(Clone, Copy)]
struct PointDistances {
    min_index: usize,
    min_distance_squared: f64,
    second_distance_squared: f64,
}

impl PointDistances {
    #[inline(always)]
    fn new() -> Self {
        Self {
            min_index: 0,
            min_distance_squared: f64::MAX,
            second_distance_squared: f64::MAX,
        }
    }

    #[inline(always)]
    fn consider(&mut self, index: usize, distance_squared: f64) {
        if distance_squared < self.min_distance_squared {
            self.second_distance_squared = self.min_distance_squared;
            self.min_distance_squared = distance_squared;
            self.min_index = index;
        } else if distance_squared < self.second_distance_squared {
            self.second_distance_squared = distance_squared;
        }
    }

    /// Order-invariant insertion with strict-`<` lowest-index tie-breaking.
    #[inline(always)]
    fn consider_unordered(&mut self, index: usize, distance_squared: f64) {
        if distance_squared < self.min_distance_squared
            || (distance_squared == self.min_distance_squared && index < self.min_index)
        {
            self.second_distance_squared = self.min_distance_squared;
            self.min_distance_squared = distance_squared;
            self.min_index = index;
        } else if distance_squared < self.second_distance_squared {
            self.second_distance_squared = distance_squared;
        }
    }
}

/// Exact distance-squared for `D = 3` or `D = 4` matching `get_distance_squared` bit-for-bit.
#[inline(always)]
pub(super) fn distance_squared_fixed<const D: usize, S: Simd>(
    simd: S,
    point: &[f64],
    centroid: &[f64],
) -> f64 {
    #[cfg(feature = "counters")]
    crate::work::count_one(&crate::work::DISTANCE_EVALUATIONS);

    if D == 3 {
        let d0 = point[0] - centroid[0];
        let d1 = point[1] - centroid[1];
        let d2 = point[2] - centroid[2];
        (d0 * d0 + d1 * d1) + d2 * d2
    } else {
        let d01 = f64x2::<S>::from_slice(simd, &point[0..2])
            - f64x2::<S>::from_slice(simd, &centroid[0..2]);
        let d23 = f64x2::<S>::from_slice(simd, &point[2..4])
            - f64x2::<S>::from_slice(simd, &centroid[2..4]);
        let acc = d01 * d01 + d23 * d23;
        acc[0] + acc[1]
    }
}

/// Triangle-inequality-pruned `point_all_centers` for `D = 3` or `D = 4`.
///
/// `neighbors` is pre-sorted ascending by distance from `current`. Evaluates
/// `neighbors[0]` first to tighten `second`, exits early once `n.dist > limit`,
/// and skips any entry inside `n.dist < lower_limit`.
#[allow(clippy::too_many_arguments)]
pub(super) fn point_all_centers_pruned_fixed<const D: usize, S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    neighbors: &[Neighbor],
    current: usize,
    current_distance_squared: f64,
    u: f64,
    state: &mut PointState,
) {
    let mut seen = PointDistances::new();
    seen.consider_unordered(current, current_distance_squared);

    let nn = neighbors[0].index as usize;
    let nn_offset = nn * D;
    let nn_d2 = distance_squared_fixed::<D, _>(simd, point, &centroids[nn_offset..nn_offset + D]);
    seen.consider_unordered(nn, nn_d2);

    let mut second = seen.second_distance_squared.sqrt();
    let mut limit = (u + second) * (1.0 + 1e-11);
    let mut lower_limit = if u > second {
        (u - second) * (1.0 - 1e-11)
    } else {
        -1.0
    };

    for &n in &neighbors[1..] {
        if n.dist > limit {
            break;
        }
        if n.dist < lower_limit {
            continue;
        }
        let j = n.index as usize;
        let j_offset = j * D;
        let d2 = distance_squared_fixed::<D, _>(simd, point, &centroids[j_offset..j_offset + D]);
        let prev_second = seen.second_distance_squared;
        seen.consider_unordered(j, d2);
        if seen.second_distance_squared < prev_second {
            second = seen.second_distance_squared.sqrt();
            limit = (u + second) * (1.0 + 1e-11);
            lower_limit = if u > second {
                (u - second) * (1.0 - 1e-11)
            } else {
                -1.0
            };
        }
    }

    state.centroid = seen.min_index as u32;
    state.upper_bound = seen.min_distance_squared.sqrt();
    state.lower_bound = second;
}

/// Triangle-inequality-pruned `point_all_centers` for arbitrary `dimensions`,
/// buffering up to 4 surviving candidates to evaluate via `distances_4x_f64`.
#[allow(clippy::too_many_arguments)]
pub(super) fn point_all_centers_pruned_generic<S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    dimensions: usize,
    neighbors: &[Neighbor],
    current: usize,
    current_distance_squared: f64,
    u: f64,
    state: &mut PointState,
) {
    let mut seen = PointDistances::new();
    seen.consider_unordered(current, current_distance_squared);

    let nn = neighbors[0].index as usize;
    let nn_d2 = get_distance_squared(simd, point, centroids, dimensions, nn * dimensions);
    seen.consider_unordered(nn, nn_d2);

    let mut second = seen.second_distance_squared.sqrt();
    let mut limit = (u + second) * (1.0 + 1e-11);
    let mut lower_limit = if u > second {
        (u - second) * (1.0 - 1e-11)
    } else {
        -1.0
    };

    let mut buf = [0usize; 4];
    let mut buf_len = 0usize;

    for &n in &neighbors[1..] {
        if n.dist > limit {
            break;
        }
        if n.dist < lower_limit {
            continue;
        }
        buf[buf_len] = n.index as usize;
        buf_len += 1;
        if buf_len == 4 {
            let offsets = [
                buf[0] * dimensions,
                buf[1] * dimensions,
                buf[2] * dimensions,
                buf[3] * dimensions,
            ];
            let d2s = distances_4x_f64(simd, point, centroids, dimensions, offsets);
            let prev_second = seen.second_distance_squared;
            for m in 0..4 {
                seen.consider_unordered(buf[m], d2s[m]);
            }
            buf_len = 0;
            if seen.second_distance_squared < prev_second {
                second = seen.second_distance_squared.sqrt();
                limit = (u + second) * (1.0 + 1e-11);
                lower_limit = if u > second {
                    (u - second) * (1.0 - 1e-11)
                } else {
                    -1.0
                };
            }
        }
    }

    for &j in &buf[..buf_len] {
        let d2 = get_distance_squared(simd, point, centroids, dimensions, j * dimensions);
        seen.consider_unordered(j, d2);
    }

    state.centroid = seen.min_index as u32;
    state.upper_bound = seen.min_distance_squared.sqrt();
    state.lower_bound = seen.second_distance_squared.sqrt();
}

/// Scans `k` centroids two at a time in `transposed` (`f64x2`) while
/// preserving the exact IEEE 754 summation order of `get_distance_squared`.
#[inline]
pub(super) fn point_all_centers_fixed<const D: usize, S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    transposed: &[f64],
    k: usize,
    known_centroid: Option<(usize, f64)>,
    state: &mut PointState,
) {
    let mut seen = PointDistances::new();
    let pairs = k / 2;

    if D == 3 {
        let px = f64x2::<S>::splat(simd, point[0]);
        let py = f64x2::<S>::splat(simd, point[1]);
        let pz = f64x2::<S>::splat(simd, point[2]);
        let row1 = k;
        let row2 = 2 * k;

        for pair in 0..pairs {
            let first = pair * 2;
            let dx = px - f64x2::<S>::from_slice(simd, &transposed[first..first + 2]);
            let dy = py - f64x2::<S>::from_slice(simd, &transposed[row1 + first..row1 + first + 2]);
            let dz = pz - f64x2::<S>::from_slice(simd, &transposed[row2 + first..row2 + first + 2]);
            // Exact IEEE 754 order matching `get_distance_squared`: (d0^2 + d1^2) + d2^2.
            let dist = (dx * dx + dy * dy) + dz * dz;
            #[cfg(feature = "counters")]
            crate::work::count(&crate::work::DISTANCE_EVALUATIONS, 2);
            seen.consider(first, dist[0]);
            seen.consider(first + 1, dist[1]);
        }
    } else if D == 4 {
        let px = f64x2::<S>::splat(simd, point[0]);
        let py = f64x2::<S>::splat(simd, point[1]);
        let pz = f64x2::<S>::splat(simd, point[2]);
        let pw = f64x2::<S>::splat(simd, point[3]);
        let row1 = k;
        let row2 = 2 * k;
        let row3 = 3 * k;

        for pair in 0..pairs {
            let first = pair * 2;
            let dx = px - f64x2::<S>::from_slice(simd, &transposed[first..first + 2]);
            let dy = py - f64x2::<S>::from_slice(simd, &transposed[row1 + first..row1 + first + 2]);
            let dz = pz - f64x2::<S>::from_slice(simd, &transposed[row2 + first..row2 + first + 2]);
            let dw = pw - f64x2::<S>::from_slice(simd, &transposed[row3 + first..row3 + first + 2]);
            // Exact IEEE 754 order matching `get_distance_squared`: (d0^2 + d2^2) + (d1^2 + d3^2).
            let dist = (dx * dx + dz * dz) + (dy * dy + dw * dw);
            #[cfg(feature = "counters")]
            crate::work::count(&crate::work::DISTANCE_EVALUATIONS, 2);
            seen.consider(first, dist[0]);
            seen.consider(first + 1, dist[1]);
        }
    } else {
        point_all_centers_generic(simd, point, centroids, D, k, known_centroid, state);
        return;
    }

    if !k.is_multiple_of(2) {
        let last = k - 1;
        let distance_squared = match known_centroid {
            Some((known, distance)) if known == last => distance,
            _ => get_distance_squared(simd, point, centroids, D, last * D),
        };
        seen.consider(last, distance_squared);
    }

    state.centroid = seen.min_index as u32;
    state.upper_bound = seen.min_distance_squared.sqrt();
    state.lower_bound = seen.second_distance_squared.sqrt();
}

pub(super) fn point_all_centers_generic<S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    dimensions: usize,
    k: usize,
    known_centroid: Option<(usize, f64)>,
    state: &mut PointState,
) {
    let mut seen = PointDistances::new();
    let mut j = 0usize;
    if known_centroid.is_none() {
        while j + 4 <= k {
            let d2s = distances_4x_f64(
                simd,
                point,
                centroids,
                dimensions,
                [
                    j * dimensions,
                    (j + 1) * dimensions,
                    (j + 2) * dimensions,
                    (j + 3) * dimensions,
                ],
            );
            for (m, &d2) in d2s.iter().enumerate() {
                seen.consider(j + m, d2);
            }
            j += 4;
        }
    }
    while j < k {
        let distance_squared = match known_centroid {
            Some((known, distance)) if known == j => distance,
            _ => get_distance_squared(simd, point, centroids, dimensions, j * dimensions),
        };
        seen.consider(j, distance_squared);
        j += 1;
    }

    state.centroid = seen.min_index as u32;
    state.upper_bound = seen.min_distance_squared.sqrt();
    state.lower_bound = seen.second_distance_squared.sqrt();
}
