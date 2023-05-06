mod kmeans_triangle;

use js_sys::{Array, Object, Function};
use wasm_bindgen::{prelude::*, JsCast};

#[wasm_bindgen]
/// Find the k-means centroids of an RGB u8 slice for color quantization.
///
/// - `rgb_slice` - Uint8Array of RGB components, where each component is a u8 value.
/// - `k >= 2` - number of clusters.
/// - `max_iter >= 1` - maximum number of iterations.
/// - `convergence_threshold > 0.0` - the threshold to determine when the centroids have converged.
///
/// This function is suitable for color quantization in image processing, where the goal is to
/// reduce the number of distinct colors in an image while preserving its overall appearance.
/// The resulting centroids represent the quantized colors.
pub fn kmeans_rgb(
    rgb_slice: Vec<u8>,
    k: usize,
    max_iter: usize,
    convergence_threshold: Option<f64>,
) -> Result<Vec<u8>, JsValue> {
    if k < 2 {
        return Err(JsValue::from_str(
            "Error: k must be greater than or equal to 2.",
        ));
    }

    if max_iter < 1 {
        return Err(JsValue::from_str(
            "Error: max_iter must be greater than or equal to 1.",
        ));
    }

    let convergence_threshold = convergence_threshold.unwrap_or(0.0);

    if convergence_threshold.is_sign_negative() {
        return Err(JsValue::from_str(
            "Error: convergence_threshold must be positive",
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
        convergence_threshold,
        rgb_slice
            .chunks_exact(3)
            .map(|x| vec![x[0] as f64, x[1] as f64, x[2] as f64])
            .collect(),
    );

    Ok(centroids.centroids
        .iter()
        .flat_map(|centroid| [centroid[0] as u8, centroid[1] as u8, centroid[2] as u8])
        .collect())
}

#[wasm_bindgen(typescript_custom_section)]
const KMEANS_TYPE: &'static str = r#"
export interface IKmeansResult {
    /** The number of iterations performed until the algorithm has converged */
    it: number;
    /** The cluster size */
    k: number;
    /** The value for each centroid of the cluster */
    centroids: number[][];
    /** The index to the centroid corresponding to each value of the data array */
    idxs: number[];
    /** Function to test new point membership */
    test: (point: number[], fnDist?: (a: number[], b: number[]) => number) => number;
}

/**
* Find the k-means centroids for any vector-space.
*
* - `data` - array of arrays of number values, where each inner array represents a point in the vector-space.
* - `k >= 2` - number of clusters.
* - `max_iter >= 1` - maximum number of iterations.
* - `convergence_threshold > 0.0` - the threshold to determine when the centroids have converged.
* @param {Array<Array<number>>} data
* @param {number} k
* @param {number} max_iter
* @param {number?} convergence_threshold
* @returns {IKmeansResult}
*/
export function kmeans(data: Array<Array<number>>, k: number, max_iter: number, convergence_threshold?: number): IKmeansResult;
"#;

#[wasm_bindgen(skip_typescript)]
/// Find the k-means centroids for any vector-space.
///
/// - `data` - array of arrays of number values, where each inner array represents a point in the vector-space.
/// - `k >= 2` - number of clusters.
/// - `max_iter >= 1` - maximum number of iterations.
/// - `convergence_threshold > 0.0` - the threshold to determine when the centroids have converged.
pub fn kmeans(
    data: Array,
    k: usize,
    max_iter: usize,
    convergence_threshold: Option<f64>,
) -> Result<Object, JsValue> {
    if k < 2 {
        return Err(JsValue::from_str(
            "Error: k must be greater than or equal to 2.",
        ));
    }

    if max_iter < 1 {
        return Err(JsValue::from_str(
            "Error: max_iter must be greater than or equal to 1.",
        ));
    }

    let convergence_threshold = convergence_threshold.unwrap_or(0.0);

    if convergence_threshold.is_sign_negative() {
        return Err(JsValue::from_str(
            "Error: convergence_threshold must be positive",
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

    let result = kmeans_triangle::hamerly_kmeans(k, max_iter, convergence_threshold, data_vec);

    let centroids_array = Array::new();
    for centroid in result.centroids.iter() {
        let centroid_array = Array::new();
        for value in centroid {
            centroid_array.push(&JsValue::from_f64(*value));
        }
        centroids_array.push(&centroid_array);
    }

    let idxs = Array::new();
    for index in result.point_centroids.iter() {
        idxs.push(&JsValue::from_f64(*index as f64));
    }

    let result_js_entries = Array::new();
    result_js_entries.push(&Array::of2(&JsValue::from_str("k"), &JsValue::from_f64(k as f64)));
    result_js_entries.push(&Array::of2(&JsValue::from_str("it"), &JsValue::from_f64(result.iterations as f64)));
    result_js_entries.push(&Array::of2(&JsValue::from_str("centroids"), &centroids_array));
    result_js_entries.push(&Array::of2(&JsValue::from_str("idxs"), &idxs));
    result_js_entries.push(&Array::of2(&JsValue::from_str("test"), &Function::new_with_args("point, fnDist", "
        if (point.length !== this.centroids.length) {
            throw new Error('Point should have the same length as centroid');
        }

        let minCentroid = 0;
        let midDist = Number.MAX_VALUE;
        let dist = fnDist ?? ((a, b) => {
            let result = 0;
            for (let i = 0; i < a.length; i++) {
                result += (a[i] - b[i]) ** 2;
            }

            return Math.sqrt(result);
        });

        for (const [i, centroid] of this.centroids.entries()) {
            const centroidDist = dist(centroid, point);
            if (centroidDist < minDist) {
                minDist = centroidDist;
                minCentroid = i;
            }
        }

        return minCentroid;
    ")));
    let result_js = Object::from_entries(&result_js_entries)?;

    Ok(result_js)
}
