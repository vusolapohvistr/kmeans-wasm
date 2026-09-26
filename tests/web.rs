//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

use kmeans_wasm::{kmeans_rgb, kmeans_rgba};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn quantizes_rgb_values() {
    let rgb = vec![
        255, 0, 0, //
        0, 255, 0, //
        0, 0, 255,
    ];

    let centroids = kmeans_rgb(rgb, 3, 100, Some(0.001)).unwrap();

    assert_eq!(centroids.len(), 9);
}

#[wasm_bindgen_test]
fn quantizes_rgba_values() {
    let rgba = vec![
        255, 0, 0, 255, //
        0, 255, 0, 255, //
        0, 0, 255, 128,
    ];

    let centroids = kmeans_rgba(rgba, 3, 100, Some(0.001)).unwrap();

    assert_eq!(centroids.len(), 12);
}

#[wasm_bindgen_test]
fn rejects_packed_slices_that_are_not_multiples_of_their_component_count() {
    let rgb_error = kmeans_rgb(vec![255, 0, 0, 0], 2, 10, None).unwrap_err();
    let rgba_error = kmeans_rgba(vec![255, 0, 0, 255, 0], 2, 10, None).unwrap_err();

    assert_eq!(
        rgb_error.as_string().as_deref(),
        Some("Error: The length of rgb_slice must be a multiple of 3.")
    );
    assert_eq!(
        rgba_error.as_string().as_deref(),
        Some("Error: The length of rgba_slice must be a multiple of 4.")
    );
}

#[wasm_bindgen_test]
fn rejects_invalid_clustering_arguments() {
    assert_eq!(
        kmeans_rgba(vec![0; 8], 1, 10, None)
            .unwrap_err()
            .as_string()
            .as_deref(),
        Some("Error: k must be greater than or equal to 2.")
    );
    assert_eq!(
        kmeans_rgba(vec![0; 8], 2, 0, None)
            .unwrap_err()
            .as_string()
            .as_deref(),
        Some("Error: max_iter must be greater than or equal to 1.")
    );
    assert_eq!(
        kmeans_rgba(vec![0; 8], 2, 10, Some(-1.0))
            .unwrap_err()
            .as_string()
            .as_deref(),
        Some("Error: convergence_threshold must be positive")
    );
}
