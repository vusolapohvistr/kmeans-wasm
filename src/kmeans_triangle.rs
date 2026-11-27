//! Hamerly k-means: public API, result shape, core loop, initialization, updates, and shared utilities.

mod core;
mod initialize;
mod update;
mod util;

pub use core::{
    HamerlyKmeansResult, hamerly_kmeans, hamerly_kmeans_dispatched,
    hamerly_kmeans_dispatched_weighted, hamerly_kmeans_weighted,
};
pub use update::Optional;
#[cfg(test)]
pub(crate) use util::get_distance_squared;
pub(crate) use util::{Random, get_centroids_with_random};

#[cfg(test)]
mod tests {
    use super::hamerly_kmeans_dispatched;

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
        let result = hamerly_kmeans_dispatched(k, 100, 0.1, &points(count, dimensions), dimensions);
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;

        let mut eat = |value: u64| {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };

        eat(result.iterations as u64);
        // The cluster count, not the flat length, so this hash stays comparable
        // with the values recorded when the result was one array per cluster.
        eat((result.centroids.len() / result.dimensions) as u64);
        for value in &result.centroids {
            eat(value.to_bits());
        }
        for index in result.assignments() {
            eat(u64::from(index));
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
        let result = hamerly_kmeans_dispatched(6, 50, 0.1, &points(500, 7), 7);

        assert_eq!(result.centroids.len(), 6 * 7);
        assert_eq!(result.dimensions, 7);
        for centroid in result.centroid_rows() {
            assert_eq!(centroid.len(), 7);
        }
    }

    #[test]
    fn returns_nothing_for_empty_input() {
        let result = hamerly_kmeans_dispatched(3, 10, 0.1, &[], 3);

        assert!(result.centroids.is_empty());
        assert!(result.assignments().is_empty());
        assert_eq!(result.iterations, 0);
    }
}
