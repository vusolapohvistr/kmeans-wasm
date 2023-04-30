mod kmeans_triangle;
mod utils;

use js_sys::Array;
use wasm_bindgen::{prelude::*, JsCast};

#[wasm_bindgen]
/// Find the k-means centroids of an RGB u8 slice for color quantization.
///
/// - `k >= 2` - number of clusters (colors).
/// - `max_iter` - maximum number of iterations.
/// - `rgb_slice` - Uint8Array of RGB components, where each component is a u8 value.
///
/// This function is suitable for color quantization in image processing, where the goal is to
/// reduce the number of distinct colors in an image while preserving its overall appearance.
/// The resulting centroids represent the quantized colors.
pub fn kmeans_rgb(k: usize, max_iter: usize, rgb_slice: Vec<u8>) -> Result<Vec<u8>, JsValue> {
    if k < 2 {
        return Err(JsValue::from_str(
            "Error: k must be greater than or equal to 2.",
        ));
    }

    if rgb_slice.len() % 3 != 0 {
        return Err(JsValue::from_str(
            "Error: The length of rgb_slice must be a multiple of 3.",
        ));
    }

    let centroids = kmeans_triangle::hamerly_kmeans(
        k,
        max_iter,
        rgb_slice
            .chunks_exact(3)
            .map(|x| vec![x[0] as f64, x[1] as f64, x[2] as f64])
            .collect(),
    );

    Ok(centroids
        .iter()
        .flat_map(|centroid| [centroid[0] as u8, centroid[1] as u8, centroid[2] as u8])
        .collect())
}

#[wasm_bindgen(typescript_custom_section)]
const KMEANS_TYPE: &'static str = r#"
/**
* Find the k-means centroids for any vector-space.
*
* - `k >= 2` - number of clusters.
* - `max_iter` - maximum number of iterations.
* - `data` - array of arrays of f64 values, where each inner array represents a point in the vector-space.
* @param {number} k
* @param {number} max_iter
* @param {Array<Array<number>>} data
* @returns {Array<Array<number>>}
*/
export function kmeans(
  k: number,
  max_iter: number,
  data: Array<Array<number>>,
): Array<Array<number>>;
"#;

#[wasm_bindgen(skip_typescript)]
/// Find the k-means centroids for any vector-space.
///
/// - `k >= 2` - number of clusters.
/// - `max_iter` - maximum number of iterations.
/// - `data` - array of arrays of f64 values, where each inner array represents a point in the vector-space.
pub fn kmeans(k: usize, max_iter: usize, data: Array) -> Result<Array, JsValue> {
    if k < 2 {
        return Err(JsValue::from_str(
            "Error: k must be greater than or equal to 2.",
        ));
    }

    let data_vec: Vec<Vec<f64>> = data
        .iter()
        .map(|point| {
            let point_array: Array = point.dyn_into().unwrap();
            let point_vec: Vec<f64> = point_array
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect();
            point_vec
        })
        .collect();

    if let Some(first_point) = data_vec.first() {
        let dimension = first_point.len();
        if data_vec.iter().any(|point| point.len() != dimension) {
            return Err(JsValue::from_str(
                "Error: All data points must have the same dimension.",
            ));
        }
    }

    let centroids = kmeans_triangle::hamerly_kmeans(k, max_iter, data_vec);

    let centroids_array = Array::new();
    for centroid in centroids {
        let centroid_array = Array::new();
        for value in centroid {
            centroid_array.push(&JsValue::from_f64(value));
        }
        centroids_array.push(&centroid_array);
    }

    Ok(centroids_array)
}
