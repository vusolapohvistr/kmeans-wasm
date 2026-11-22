//! Assignment phase: nearest-centroid scans for the moving centroids used by
//! e-means.

use fearless_simd::{Simd, SimdBase, f64x2};

use crate::kmeans_triangle::get_distance_squared;

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
pub(super) fn nearest_centroid<S: Simd>(
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
///
/// The three- and four-component cases are specialized so the inner loop is
/// fully unrolled and reads the point values as constants; the generic tail
/// stays for every other dimension.
#[inline]
pub(super) fn nearest_centroid_fixed<const D: usize, S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    transposed: &[f64],
    k: usize,
) -> usize {
    if D == 3 {
        nearest_centroid_fixed_3(simd, point, centroids, transposed, k)
    } else if D == 4 {
        nearest_centroid_fixed_4(simd, point, centroids, transposed, k)
    } else {
        nearest_centroid_fixed_generic::<D, _>(simd, point, centroids, transposed, k)
    }
}

#[inline]
fn nearest_centroid_fixed_3<S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    transposed: &[f64],
    k: usize,
) -> usize {
    let point_x = f64x2::<S>::splat(simd, point[0]);
    let point_y = f64x2::<S>::splat(simd, point[1]);
    let point_z = f64x2::<S>::splat(simd, point[2]);
    let mut nearest = 0;
    let mut nearest_distance_squared = f64::MAX;

    for pair in 0..k / 2 {
        let first_cluster = pair * 2;
        let cx = f64x2::<S>::from_slice(simd, &transposed[first_cluster..first_cluster + 2]);
        let dx = point_x - cx;
        let offset = k + first_cluster;
        let cy = f64x2::<S>::from_slice(simd, &transposed[offset..offset + 2]);
        let dy = point_y - cy;
        let offset = 2 * k + first_cluster;
        let cz = f64x2::<S>::from_slice(simd, &transposed[offset..offset + 2]);
        let dz = point_z - cz;

        let accumulator = dx * dx + dy * dy + dz * dz;
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
        let distance_squared = get_distance_squared(simd, point, centroids, 3, (k - 1) * 3);
        if distance_squared < nearest_distance_squared {
            nearest = k - 1;
        }
    }

    nearest
}

#[inline]
fn nearest_centroid_fixed_4<S: Simd>(
    simd: S,
    point: &[f64],
    centroids: &[f64],
    transposed: &[f64],
    k: usize,
) -> usize {
    let point_x = f64x2::<S>::splat(simd, point[0]);
    let point_y = f64x2::<S>::splat(simd, point[1]);
    let point_z = f64x2::<S>::splat(simd, point[2]);
    let point_w = f64x2::<S>::splat(simd, point[3]);
    let mut nearest = 0;
    let mut nearest_distance_squared = f64::MAX;

    for pair in 0..k / 2 {
        let first_cluster = pair * 2;
        let cx = f64x2::<S>::from_slice(simd, &transposed[first_cluster..first_cluster + 2]);
        let dx = point_x - cx;
        let offset = k + first_cluster;
        let cy = f64x2::<S>::from_slice(simd, &transposed[offset..offset + 2]);
        let dy = point_y - cy;
        let offset = 2 * k + first_cluster;
        let cz = f64x2::<S>::from_slice(simd, &transposed[offset..offset + 2]);
        let dz = point_z - cz;
        let offset = 3 * k + first_cluster;
        let cw = f64x2::<S>::from_slice(simd, &transposed[offset..offset + 2]);
        let dw = point_w - cw;

        let accumulator = dx * dx + dy * dy + dz * dz + dw * dw;
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
        let distance_squared = get_distance_squared(simd, point, centroids, 4, (k - 1) * 4);
        if distance_squared < nearest_distance_squared {
            nearest = k - 1;
        }
    }

    nearest
}

#[inline]
fn nearest_centroid_fixed_generic<const D: usize, S: Simd>(
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
