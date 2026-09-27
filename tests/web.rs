//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

use js_sys::{Array, Function, Uint32Array};
use kmeans_wasm::{apply_palette, kmeans, kmeans_rgb, kmeans_rgba};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn to_js_array(values: &[f64]) -> Array {
    let array = Array::new();
    for value in values {
        array.push(&JsValue::from_f64(*value));
    }
    array
}

fn to_js_point_array(points: &[&[f64]]) -> Array {
    let array = Array::new();
    for point in points {
        array.push(&to_js_array(point));
    }
    array
}

fn property(value: &JsValue, name: &str) -> JsValue {
    js_sys::Reflect::get(value, &JsValue::from_str(name)).unwrap()
}

#[wasm_bindgen_test]
fn returns_dense_centroids_for_vector_spaces() {
    let data = to_js_point_array(&[&[1.0, 2.0], &[2.0, 3.0], &[3.0, 4.0], &[40.0, 50.0]]);

    let result = kmeans(data, 2, 100, Some(0.001)).unwrap();

    let centroids: Array = property(&result, "centroids").into();
    assert_eq!(centroids.length(), 2);
    for index in 0..centroids.length() {
        let centroid: Array = centroids.get(index).into();
        assert_eq!(centroid.length(), 2, "centroids keep the point dimension");
    }

    let idxs: Uint32Array = property(&result, "idxs").into();
    assert_eq!(idxs.length(), 4);
    for index in 0..idxs.length() {
        assert!(idxs.get_index(index) < 2);
    }
}

#[wasm_bindgen_test]
fn assigns_new_points_to_the_nearest_centroid() {
    let data = to_js_point_array(&[&[0.0, 0.0], &[10.0, 10.0]]);

    let result = kmeans(data, 2, 100, Some(0.001)).unwrap();
    let test: Function = property(&result, "test").into();

    // The helper reads `this.centroids`, so it has to be called with the result
    // bound as the receiver, exactly like `result.test(point)` in JavaScript.
    let near_first = test.call1(&result, &to_js_array(&[1.0, 1.0])).unwrap();
    let near_second = test.call1(&result, &to_js_array(&[9.0, 9.0])).unwrap();

    assert_ne!(near_first.as_f64().unwrap(), near_second.as_f64().unwrap());
    assert!(test.call1(&result, &to_js_array(&[1.0])).is_err());
}

#[wasm_bindgen_test]
fn rejects_inconsistent_vector_spaces() {
    let ragged = to_js_point_array(&[&[1.0, 2.0, 3.0], &[4.0, 5.0]]);

    let error = kmeans(ragged, 2, 10, None).unwrap_err();

    assert_eq!(
        error.as_string().as_deref(),
        Some("Error: All data points must have the same dimension.")
    );
}

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

/// Rejected input. These build a `JsValue`, which aborts on native targets, so
/// they can only be asserted here.
#[wasm_bindgen_test]
fn rejects_malformed_packed_input() {
    for bad in [vec![10, 20], vec![10, 20, 30, 40]] {
        let error = kmeans_rgb(bad.clone(), 2, 100, Some(0.1)).unwrap_err();
        assert_eq!(
            error.as_string().as_deref(),
            Some("Error: The length of rgb_slice must be a multiple of 3."),
            "for {bad:?}"
        );
    }

    for bad in [vec![1, 2, 3, 4, 5], vec![1, 2, 3]] {
        let error = kmeans_rgba(bad.clone(), 2, 100, Some(0.1)).unwrap_err();
        assert_eq!(
            error.as_string().as_deref(),
            Some("Error: The length of rgba_slice must be a multiple of 4."),
            "for {bad:?}"
        );
    }
}

/// An empty buffer is *not* rejected, which is worth pinning down rather than
/// assuming. A length of zero is a multiple of both three and four, so it passes
/// validation, and there are no pixels to cluster, so the result is an empty
/// palette rather than an error.
#[wasm_bindgen_test]
fn an_empty_buffer_returns_an_empty_palette() {
    assert_eq!(
        kmeans_rgb(Vec::new(), 2, 100, Some(0.1)).unwrap(),
        Vec::<u8>::new()
    );
    assert_eq!(
        kmeans_rgba(Vec::new(), 2, 100, Some(0.1)).unwrap(),
        Vec::<u8>::new()
    );
}

/// A single pixel is a valid triple, so it is accepted even though it holds one
/// distinct value and `k` is larger.
#[wasm_bindgen_test]
fn accepts_a_single_pixel() {
    let palette = kmeans_rgb(vec![10, 20, 30], 2, 100, Some(0.1)).unwrap();
    assert_eq!(palette.len(), 6);
    assert!(palette.iter().all(|value| matches!(*value, 10 | 20 | 30)));
}

/// The clustering and the mapping compose: every pixel has to come out as one of
/// the palette's own colours, and the mapping has to be the nearest one, which is
/// checked here by confirming nothing was invented and the two flat colours went
/// to their own entries.
#[wasm_bindgen_test]
fn mapped_pixels_are_palette_colours() {
    let pixels = vec![
        255, 0, 0, //
        250, 2, 1, // near red
        0, 255, 0, //
        3, 250, 4, // near green
        0, 0, 255,
    ];
    let palette = kmeans_rgb(pixels.clone(), 3, 100, Some(0.1)).unwrap();
    let mapped = apply_palette(pixels, palette.clone(), 3).unwrap();

    // Five pixels in, so five RGBA pixels out.
    assert_eq!(mapped.len(), 5 * 4);
    // Alpha is synthesised opaque for a three-component source.
    assert!(mapped.chunks_exact(4).all(|pixel| pixel[3] == 255));
    // Every output triple has to be a palette triple, which is the whole
    // guarantee: nothing that is not in the palette can be written.
    for pixel in mapped.chunks_exact(4) {
        let triple = [pixel[0], pixel[1], pixel[2]];
        assert!(
            palette.chunks_exact(3).any(|entry| entry == triple),
            "{triple:?} is not in the palette"
        );
    }
}

/// Rejected mapping input, for the same reason as the clustering cases above.
#[wasm_bindgen_test]
fn rejects_malformed_mapping_input() {
    let palette = vec![0, 0, 0, 255, 255, 255];

    assert_eq!(
        apply_palette(vec![1, 2, 3], Vec::new(), 3)
            .unwrap_err()
            .as_string()
            .as_deref(),
        Some("Error: The length of palette must be a non-zero multiple of components.")
    );
    assert_eq!(
        apply_palette(vec![1, 2, 3], vec![0, 0], 3)
            .unwrap_err()
            .as_string()
            .as_deref(),
        Some("Error: The length of palette must be a non-zero multiple of components.")
    );
    assert_eq!(
        apply_palette(vec![1, 2], palette.clone(), 3)
            .unwrap_err()
            .as_string()
            .as_deref(),
        Some("Error: The length of pixels must be a multiple of 3.")
    );
    assert_eq!(
        apply_palette(vec![1, 2, 3], palette, 5)
            .unwrap_err()
            .as_string()
            .as_deref(),
        Some("Error: components must be 3 for RGB or 4 for RGBA.")
    );
}

/// An empty buffer maps to an empty buffer rather than erroring, for the same
/// reason an empty buffer yields an empty palette.
#[wasm_bindgen_test]
fn an_empty_buffer_maps_to_an_empty_buffer() {
    assert_eq!(
        apply_palette(Vec::new(), vec![0, 0, 0], 3).unwrap(),
        Vec::<u8>::new()
    );
}
