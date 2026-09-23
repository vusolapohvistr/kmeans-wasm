//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

use kmeans_wasm::kmeans_rgb;
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
