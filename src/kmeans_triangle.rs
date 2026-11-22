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
pub(crate) use util::{Random, get_centroids_with_random, get_distance_squared};
