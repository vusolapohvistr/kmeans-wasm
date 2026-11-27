//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

use js_sys::{
    Array, Error, Float32Array, Float64Array, Function, RangeError, TypeError, Uint32Array,
};
use kmeans_wasm::{
    apply_palette, apply_palette_dithered, apply_palette_indices, emeans, emeans_f32, emeans_f64,
    emeans_rgb, emeans_rgba, kmeans, kmeans_f32, kmeans_f64, kmeans_rgb, kmeans_rgba,
};
use wasm_bindgen::{JsCast, JsValue};
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

fn error_message(value: JsValue) -> String {
    assert!(
        value.is_instance_of::<Error>(),
        "expected Error instance, got {value:?}"
    );
    let err: Error = value.dyn_into().unwrap();
    err.message().into()
}

#[wasm_bindgen_test]
fn returns_dense_centroids_for_vector_spaces() {
    let data = to_js_point_array(&[&[1.0, 2.0], &[2.0, 3.0], &[3.0, 4.0], &[40.0, 50.0]]);

    let result = kmeans(data, 2, 100, Some(0.001), None).unwrap();

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

    let result = kmeans(data, 2, 100, Some(0.001), None).unwrap();
    let test: Function = property(&result, "test").into();

    // The helper reads `this.centroids`, so it has to be called with the result
    // bound as the receiver, exactly like `result.test(point)` in JavaScript.
    let near_first = test.call1(&result, &to_js_array(&[1.0, 1.0])).unwrap();
    let near_second = test.call1(&result, &to_js_array(&[9.0, 9.0])).unwrap();

    assert_ne!(near_first.as_f64().unwrap(), near_second.as_f64().unwrap());
    assert!(test.call1(&result, &to_js_array(&[1.0])).is_err());
}

#[wasm_bindgen_test]
fn rejects_inconsistent_and_malformed_vector_spaces() {
    let ragged = to_js_point_array(&[&[1.0, 2.0, 3.0], &[4.0, 5.0]]);
    let error = kmeans(ragged, 2, 10, None, None).unwrap_err();
    assert!(error.is_instance_of::<RangeError>());
    assert_eq!(
        error_message(error),
        "Error: All data points must have the same dimension."
    );

    let non_array = Array::new();
    non_array.push(&JsValue::from_f64(1.0));
    non_array.push(&to_js_array(&[2.0]));
    let type_err = kmeans(non_array, 2, 10, None, None).unwrap_err();
    assert!(type_err.is_instance_of::<TypeError>());

    let with_nan = to_js_point_array(&[&[1.0, f64::NAN], &[2.0, 3.0]]);
    let nan_err = kmeans(with_nan, 2, 10, None, None).unwrap_err();
    assert!(nan_err.is_instance_of::<RangeError>());

    let with_inf = to_js_point_array(&[&[1.0, f64::INFINITY], &[2.0, 3.0]]);
    let inf_err = emeans(with_inf, 2, Some(2), None, None).unwrap_err();
    assert!(inf_err.is_instance_of::<RangeError>());
}

#[wasm_bindgen_test]
fn quantizes_rgb_values() {
    let rgb = [
        255, 0, 0, //
        0, 255, 0, //
        0, 0, 255,
    ];

    let centroids = kmeans_rgb(&rgb, 3, 100, Some(0.001), None).unwrap();

    assert_eq!(centroids.len(), 9);
}

#[wasm_bindgen_test]
fn quantizes_rgba_values() {
    let rgba = [
        255, 0, 0, 255, //
        0, 255, 0, 255, //
        0, 0, 255, 128,
    ];

    let centroids = kmeans_rgba(&rgba, 3, 100, Some(0.001), None).unwrap();

    assert_eq!(centroids.len(), 12);
}

#[wasm_bindgen_test]
fn rejects_packed_slices_that_are_not_multiples_of_their_component_count() {
    let rgb_error = kmeans_rgb(&[255, 0, 0, 0], 2, 10, None, None).unwrap_err();
    let rgba_error = kmeans_rgba(&[255, 0, 0, 255, 0], 2, 10, None, None).unwrap_err();

    assert_eq!(
        error_message(rgb_error),
        "Error: The length of rgb_slice must be a multiple of 3."
    );
    assert_eq!(
        error_message(rgba_error),
        "Error: The length of rgba_slice must be a multiple of 4."
    );
}

#[wasm_bindgen_test]
fn rejects_invalid_clustering_arguments() {
    assert_eq!(
        error_message(kmeans_rgba(&[0; 8], 1, 10, None, None).unwrap_err()),
        "Error: k must be greater than or equal to 2."
    );
    assert_eq!(
        error_message(kmeans_rgba(&[0; 8], 2, 0, None, None).unwrap_err()),
        "Error: max_iter must be greater than or equal to 1."
    );
    assert_eq!(
        error_message(kmeans_rgba(&[0; 8], 2, 10, Some(-1.0), None).unwrap_err()),
        "Error: convergence_threshold must be positive"
    );
    assert_eq!(
        error_message(kmeans_rgba(&[0; 8], 2, 10, Some(f64::NAN), None).unwrap_err()),
        "Error: convergence_threshold must be positive"
    );
    assert_eq!(
        error_message(kmeans_rgba(&[0; 8], 2, 10, Some(f64::INFINITY), None).unwrap_err()),
        "Error: convergence_threshold must be positive"
    );
}

/// Rejected input. These build a `JsValue`, which aborts on native targets, so
/// they can only be asserted here.
#[wasm_bindgen_test]
fn rejects_malformed_packed_input() {
    for bad in [vec![10, 20], vec![10, 20, 30, 40]] {
        let error = kmeans_rgb(&bad, 2, 100, Some(0.1), None).unwrap_err();
        assert_eq!(
            error_message(error),
            "Error: The length of rgb_slice must be a multiple of 3.",
            "for {bad:?}"
        );
    }

    for bad in [vec![1, 2, 3, 4, 5], vec![1, 2, 3]] {
        let error = kmeans_rgba(&bad, 2, 100, Some(0.1), None).unwrap_err();
        assert_eq!(
            error_message(error),
            "Error: The length of rgba_slice must be a multiple of 4.",
            "for {bad:?}"
        );
    }
}

#[wasm_bindgen_test]
fn an_empty_buffer_returns_an_empty_palette() {
    assert_eq!(
        kmeans_rgb(&[], 2, 100, Some(0.1), None).unwrap(),
        Vec::<u8>::new()
    );
    assert_eq!(
        kmeans_rgba(&[], 2, 100, Some(0.1), None).unwrap(),
        Vec::<u8>::new()
    );
}

#[wasm_bindgen_test]
fn accepts_a_single_pixel() {
    let palette = kmeans_rgb(&[10, 20, 30], 2, 100, Some(0.1), None).unwrap();
    assert_eq!(palette.len(), 6);
    assert!(palette.iter().all(|value| matches!(*value, 10 | 20 | 30)));
}

#[wasm_bindgen_test]
fn mapped_pixels_are_palette_colours() {
    let pixels = vec![
        255, 0, 0, //
        250, 2, 1, // near red
        0, 255, 0, //
        3, 250, 4, // near green
        0, 0, 255,
    ];
    let palette = kmeans_rgb(&pixels, 3, 100, Some(0.1), Some(7)).unwrap();
    let mapped = apply_palette(&pixels, &palette, 3).unwrap();
    let indices = apply_palette_indices(&pixels, &palette, 3).unwrap();
    let dithered = apply_palette_dithered(&pixels, 5, 1, &palette, 3).unwrap();

    assert_eq!(mapped.len(), 5 * 4);
    assert_eq!(indices.len(), 5);
    assert_eq!(dithered.len(), 5 * 4);
    let (mapped_rows, _) = mapped.as_chunks::<4>();
    let (dithered_rows, _) = dithered.as_chunks::<4>();
    let (palette_rows, _) = palette.as_chunks::<3>();
    assert!(mapped_rows.iter().all(|pixel| pixel[3] == 255));
    assert!(dithered_rows.iter().all(|pixel| pixel[3] == 255));
    for (idx, pixel) in mapped_rows.iter().enumerate() {
        let triple = [pixel[0], pixel[1], pixel[2]];
        assert!(
            palette_rows.contains(&triple),
            "{triple:?} is not in the palette"
        );
        let entry_idx = indices[idx] as usize;
        assert_eq!(&palette[entry_idx * 3..entry_idx * 3 + 3], &triple);
    }
}

#[wasm_bindgen_test]
fn rejects_malformed_mapping_input() {
    let palette = vec![0, 0, 0, 255, 255, 255];

    assert_eq!(
        error_message(apply_palette(&[1, 2, 3], &[], 3).unwrap_err()),
        "Error: The length of palette must be a non-zero multiple of components."
    );
    assert_eq!(
        error_message(apply_palette(&[1, 2, 3], &[0, 0], 3).unwrap_err()),
        "Error: The length of palette must be a non-zero multiple of components."
    );
    assert_eq!(
        error_message(apply_palette(&[1, 2], &palette, 3).unwrap_err()),
        "Error: The length of pixels must be a multiple of 3."
    );
    assert_eq!(
        error_message(apply_palette(&[1, 2, 3], &palette, 5).unwrap_err()),
        "Error: components must be 3 for RGB or 4 for RGBA."
    );
    assert_eq!(
        error_message(apply_palette_dithered(&[1, 2, 3, 4, 5, 6], 3, 1, &palette, 3).unwrap_err()),
        "Error: pixels length must equal width * height * components."
    );
}

#[wasm_bindgen_test]
fn an_empty_buffer_maps_to_an_empty_buffer() {
    assert_eq!(apply_palette(&[], &[0, 0, 0], 3).unwrap(), Vec::<u8>::new());
    assert_eq!(
        apply_palette_indices(&[], &[0, 0, 0], 3).unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        apply_palette_dithered(&[], 0, 0, &[0, 0, 0], 3).unwrap(),
        Vec::<u8>::new()
    );
}

#[wasm_bindgen_test]
fn emeans_quantizes_packed_buffers() {
    let rgb = [
        255, 0, 0, //
        0, 255, 0, //
        0, 0, 255, //
        255, 255, 0, //
        0, 255, 255, //
        255, 0, 255,
    ];
    let rgba = [
        255, 0, 0, 255, //
        0, 255, 0, 255, //
        0, 0, 255, 128, //
        255, 255, 255, 0,
    ];

    assert_eq!(emeans_rgb(&rgb, 3, Some(5), None, None).unwrap().len(), 9);
    assert_eq!(
        emeans_rgba(&rgba, 4, Some(5), Some(2), None).unwrap().len(),
        16
    );
    assert_eq!(
        emeans_rgb(&[0; 3 * 64], 2, None, None, None).unwrap().len(),
        6
    );
}

#[wasm_bindgen_test]
fn explicit_seed_is_deterministic_on_wasm() {
    let mut rgb = Vec::new();
    for i in 0..500u16 {
        rgb.push((i % 255) as u8);
        rgb.push(((i * 7) % 255) as u8);
        rgb.push(((i * 13) % 255) as u8);
    }
    let a = kmeans_rgb(&rgb, 8, 20, Some(0.1), Some(1337)).unwrap();
    let b = kmeans_rgb(&rgb, 8, 20, Some(0.1), Some(1337)).unwrap();
    assert_eq!(a, b);

    let ea = emeans_rgb(&rgb, 8, Some(3), None, Some(1337)).unwrap();
    let eb = emeans_rgb(&rgb, 8, Some(3), None, Some(1337)).unwrap();
    assert_eq!(ea, eb);
}

#[wasm_bindgen_test]
fn rejects_invalid_emeans_arguments() {
    assert_eq!(
        error_message(emeans_rgb(&[0; 6], 1, Some(5), None, None).unwrap_err()),
        "Error: k must be greater than or equal to 2."
    );
    assert_eq!(
        error_message(emeans_rgb(&[0; 6], 2, Some(0), None, None).unwrap_err()),
        "Error: epochs must be greater than or equal to 1."
    );
    assert_eq!(
        error_message(emeans_rgba(&[0; 8], 2, Some(5), Some(0), None).unwrap_err()),
        "Error: batch_size must be greater than or equal to 1."
    );
    assert_eq!(
        error_message(emeans_rgb(&[0; 4], 2, Some(5), None, None).unwrap_err()),
        "Error: The length of rgb_slice must be a multiple of 3."
    );
    assert_eq!(
        error_message(emeans_rgba(&[0; 6], 2, Some(5), None, None).unwrap_err()),
        "Error: The length of rgba_slice must be a multiple of 4."
    );

    let ragged = to_js_point_array(&[&[1.0, 2.0, 3.0], &[4.0, 5.0]]);
    assert_eq!(
        error_message(emeans(ragged, 2, Some(5), None, None).unwrap_err()),
        "Error: All data points must have the same dimension."
    );
    assert_eq!(
        error_message(
            emeans(to_js_point_array(&[&[1.0], &[2.0]]), 1, Some(5), None, None).unwrap_err()
        ),
        "Error: k must be greater than or equal to 2."
    );
    assert_eq!(
        error_message(
            emeans(to_js_point_array(&[&[1.0], &[2.0]]), 2, Some(0), None, None).unwrap_err()
        ),
        "Error: epochs must be greater than or equal to 1."
    );
    assert_eq!(
        error_message(
            emeans(
                to_js_point_array(&[&[1.0], &[2.0]]),
                2,
                Some(5),
                Some(0),
                None
            )
            .unwrap_err()
        ),
        "Error: batch_size must be greater than or equal to 1."
    );
}

#[wasm_bindgen_test]
fn emeans_returns_a_dense_vector_result() {
    let data = to_js_point_array(&[
        &[0.0, 0.0],   //
        &[0.0, 1.0],   //
        &[10.0, 10.0], //
        &[10.0, 11.0],
    ]);

    let result = emeans(data, 2, Some(3), Some(2), None).unwrap();

    assert_eq!(property(&result, "k").as_f64(), Some(2.0));
    assert_eq!(property(&result, "it").as_f64(), Some(3.0));

    let centroids: Array = property(&result, "centroids").into();
    assert_eq!(centroids.length(), 2);
    for index in 0..centroids.length() {
        let centroid: Array = centroids.get(index).into();
        assert_eq!(centroid.length(), 2);
        assert!(
            centroid.iter().all(|value| value.as_f64().is_some()),
            "centroid {index} should be dense"
        );
    }

    let idxs: Uint32Array = property(&result, "idxs").into();
    assert_eq!(idxs.length(), 4);
    for index in 0..idxs.length() {
        assert!(idxs.get_index(index) < 2);
    }

    let test: Function = property(&result, "test").into();
    assert!(test.call1(&result, &to_js_array(&[0.5, 0.5])).is_ok());

    let defaulted = emeans(to_js_point_array(&[&[0.0], &[10.0]]), 2, None, None, None).unwrap();
    assert_eq!(property(&defaulted, "it").as_f64(), Some(5.0));
}

#[wasm_bindgen_test]
fn flat_typed_array_entry_points_work_and_validate() {
    let data64 = [0.0f64, 0.0, 0.0, 1.0, 10.0, 10.0, 10.0, 11.0];
    let res64 = kmeans_f64(&data64, 2, 2, 50, Some(0.001), Some(42)).unwrap();
    assert_eq!(property(&res64, "k").as_f64(), Some(2.0));
    assert_eq!(property(&res64, "dimensions").as_f64(), Some(2.0));
    let c64: Float64Array = property(&res64, "centroids").into();
    assert_eq!(c64.length(), 4);
    let idx64: Uint32Array = property(&res64, "idxs").into();
    assert_eq!(idx64.length(), 4);

    let data32 = [0.0f32, 0.0, 0.0, 1.0, 10.0, 10.0, 10.0, 11.0];
    let res32 = kmeans_f32(&data32, 2, 2, 50, Some(0.001), Some(42)).unwrap();
    let c32: Float32Array = property(&res32, "centroids").into();
    assert_eq!(c32.length(), 4);

    let eres64 = emeans_f64(&data64, 2, 2, Some(5), None, Some(42)).unwrap();
    assert_eq!(property(&eres64, "it").as_f64(), Some(5.0));
    let ec64: Float64Array = property(&eres64, "centroids").into();
    assert_eq!(ec64.length(), 4);

    let eres32 = emeans_f32(&data32, 2, 2, Some(5), None, Some(42)).unwrap();
    let ec32: Float32Array = property(&eres32, "centroids").into();
    assert_eq!(ec32.length(), 4);

    assert!(kmeans_f64(&[1.0, 2.0, 3.0], 2, 2, 10, None, None).is_err());
    assert!(kmeans_f64(&[1.0, f64::NAN], 1, 2, 10, None, None).is_err());
    assert!(emeans_f32(&[1.0, f32::INFINITY], 1, 2, None, None, None).is_err());
}
