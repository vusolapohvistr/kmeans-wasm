//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

use js_sys::{Array, Function, Uint32Array};
use kmeans_wasm::{kmeans, kmeans_rgb, kmeans_rgba};
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

    let near_first = test
        .call1(&JsValue::NULL, &to_js_array(&[1.0, 1.0]))
        .unwrap();
    let near_second = test
        .call1(&JsValue::NULL, &to_js_array(&[9.0, 9.0]))
        .unwrap();

    assert_ne!(near_first.as_f64().unwrap(), near_second.as_f64().unwrap());
    assert!(test.call1(&JsValue::NULL, &to_js_array(&[1.0])).is_err());
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
