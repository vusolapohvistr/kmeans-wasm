pub mod emeans;
pub mod kmeans_triangle;
pub mod packed_histogram;
#[cfg(feature = "counters")]
pub mod work;

use js_sys::{
    Array, Float32Array, Float64Array, Function, Object, RangeError, Reflect, TypeError,
    Uint32Array,
};
use wasm_bindgen::{JsCast, prelude::*};

/// Number of vector components clustered by [`kmeans_rgb`].
const RGB_COMPONENTS: usize = 3;
/// Number of vector components clustered by [`kmeans_rgba`].
const RGBA_COMPONENTS: usize = 4;

fn range_error(message: &str) -> JsValue {
    RangeError::new(message).into()
}

fn type_error(message: &str) -> JsValue {
    TypeError::new(message).into()
}

#[wasm_bindgen(inline_js = r#"
export function make_test_fn() {
    return function test(point, fnDist) {
        const centroids = this.centroids;
        if (!centroids || centroids.length === 0 || !point || point.length !== centroids[0].length) {
            throw new RangeError("Point should have the same length as centroid");
        }

        let minCentroid = 0;
        let minDist = Infinity;

        if (fnDist) {
            for (let i = 0; i < centroids.length; i += 1) {
                const d = fnDist(centroids[i], point);
                if (d < minDist) {
                    minDist = d;
                    minCentroid = i;
                }
            }
        } else {
            const dims = point.length;
            for (let i = 0; i < centroids.length; i += 1) {
                const centroid = centroids[i];
                let sum = 0;
                for (let d = 0; d < dims; d += 1) {
                    const diff = centroid[d] - point[d];
                    sum += diff * diff;
                }
                if (sum < minDist) {
                    minDist = sum;
                    minCentroid = i;
                }
            }
        }

        return minCentroid;
    };
}
"#)]
extern "C" {
    fn make_test_fn() -> Function;
}

fn validate_kmeans_common(
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
) -> Result<(), JsValue> {
    if k < 2 {
        return Err(range_error("Error: k must be greater than or equal to 2."));
    }

    if max_iter < 1 {
        return Err(range_error(
            "Error: max_iter must be greater than or equal to 1.",
        ));
    }

    if !convergence_threshold.is_finite() || convergence_threshold < 0.0 {
        return Err(range_error("Error: convergence_threshold must be positive"));
    }

    Ok(())
}

fn validate_arguments(
    slice_name: &str,
    slice_len: usize,
    components: usize,
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
) -> Result<(), JsValue> {
    validate_kmeans_common(k, max_iter, convergence_threshold)?;

    if !slice_len.is_multiple_of(components) {
        return Err(range_error(&format!(
            "Error: The length of {slice_name} must be a multiple of {components}."
        )));
    }

    Ok(())
}

fn validate_emeans_common(
    k: usize,
    epochs: usize,
    batch_size: Option<usize>,
) -> Result<(), JsValue> {
    if k < 2 {
        return Err(range_error("Error: k must be greater than or equal to 2."));
    }

    if epochs < 1 {
        return Err(range_error(
            "Error: epochs must be greater than or equal to 1.",
        ));
    }

    if batch_size.is_some_and(|size| size < 1) {
        return Err(range_error(
            "Error: batch_size must be greater than or equal to 1.",
        ));
    }

    Ok(())
}

fn validate_emeans_arguments(
    slice_name: &str,
    slice_len: usize,
    components: usize,
    k: usize,
    epochs: usize,
    batch_size: Option<usize>,
) -> Result<(), JsValue> {
    validate_emeans_common(k, epochs, batch_size)?;

    if !slice_len.is_multiple_of(components) {
        return Err(range_error(&format!(
            "Error: The length of {slice_name} must be a multiple of {components}."
        )));
    }

    Ok(())
}

fn validate_palette_inputs(
    pixels_len: usize,
    palette_len: usize,
    components: usize,
) -> Result<(), JsValue> {
    if !matches!(components, RGB_COMPONENTS | RGBA_COMPONENTS) {
        return Err(range_error(
            "Error: components must be 3 for RGB or 4 for RGBA.",
        ));
    }
    if !pixels_len.is_multiple_of(components) {
        return Err(range_error(&format!(
            "Error: The length of pixels must be a multiple of {components}."
        )));
    }
    if palette_len == 0 || !palette_len.is_multiple_of(components) {
        return Err(range_error(
            "Error: The length of palette must be a non-zero multiple of components.",
        ));
    }
    Ok(())
}

fn validate_flat_buffer_shape(data_len: usize, dimensions: usize) -> Result<(), JsValue> {
    if dimensions < 1 {
        return Err(range_error(
            "Error: dimensions must be greater than or equal to 1.",
        ));
    }
    if !data_len.is_multiple_of(dimensions) {
        return Err(range_error(
            "Error: The length of data must be a multiple of dimensions.",
        ));
    }
    Ok(())
}

/// The default convergence threshold for the packed entry points.
///
/// The result is a `u8`, so there is nothing to gain from resolving centroids
/// more finely than a component can represent. A threshold of 0.1 means a round
/// is only worth running if it moves the centroids by more than a tenth of an
/// 8-bit level, which is where a colour can still change.
const PACKED_CONVERGENCE_THRESHOLD: f64 = 0.1;

fn quantize_packed_colors(
    slice: &[u8],
    components: usize,
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
    seed: Option<u32>,
) -> Vec<u8> {
    let collapsed = packed_histogram::collapse(slice, components);
    let widened;
    let (points, weights): (&[f64], Option<&[f64]>) = match collapsed.as_ref() {
        Some(found) => (found.points.as_slice(), Some(found.weights.as_slice())),
        None => {
            let mut buffer = Vec::with_capacity(slice.len());
            for value in slice {
                buffer.push(f64::from(*value));
            }
            widened = buffer;
            (widened.as_slice(), None)
        }
    };

    let centroids = kmeans_triangle::hamerly_kmeans_dispatched_weighted(
        k,
        max_iter,
        convergence_threshold,
        points,
        components,
        kmeans_triangle::Optional {
            weights,
            seed,
            ..Default::default()
        },
    );

    centroids
        .centroids
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect()
}

fn quantize_packed_colors_emeans(
    slice: &[u8],
    components: usize,
    k: usize,
    epochs: usize,
    batch_size: Option<usize>,
    seed: Option<u32>,
) -> Vec<u8> {
    let collapsed = packed_histogram::collapse(slice, components);
    let widened;
    let (points, weights): (&[f64], Option<&[f64]>) = match collapsed.as_ref() {
        Some(found) => (found.points.as_slice(), Some(found.weights.as_slice())),
        None => {
            let mut buffer = Vec::with_capacity(slice.len());
            for value in slice {
                buffer.push(f64::from(*value));
            }
            widened = buffer;
            (widened.as_slice(), None)
        }
    };

    let result = emeans::emeans_dispatched(
        k,
        epochs,
        points,
        components,
        emeans::EmeansOptions {
            weights,
            batch_size,
            seed,
            use_f32_assignment: true,
            use_dot_product: true,
            ..Default::default()
        },
    );

    result
        .centroids
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[wasm_bindgen(skip_typescript)]
/// Find the e-means centroids of an RGB u8 slice for color quantization.
pub fn emeans_rgb(
    rgb_slice: &[u8],
    k: usize,
    epochs: Option<u32>,
    batch_size: Option<u32>,
    seed: Option<u32>,
) -> Result<Vec<u8>, JsValue> {
    let epochs = epochs.map_or(emeans::DEFAULT_EPOCHS, |value| value as usize);
    let batch_size = batch_size.map(|value| value as usize);

    validate_emeans_arguments(
        "rgb_slice",
        rgb_slice.len(),
        RGB_COMPONENTS,
        k,
        epochs,
        batch_size,
    )?;

    Ok(quantize_packed_colors_emeans(
        rgb_slice,
        RGB_COMPONENTS,
        k,
        epochs,
        batch_size,
        seed,
    ))
}

#[wasm_bindgen(skip_typescript)]
/// Find the k-means centroids of an RGB u8 slice for color quantization.
pub fn kmeans_rgb(
    rgb_slice: &[u8],
    k: usize,
    max_iter: usize,
    convergence_threshold: Option<f64>,
    seed: Option<u32>,
) -> Result<Vec<u8>, JsValue> {
    let convergence_threshold = convergence_threshold.unwrap_or(PACKED_CONVERGENCE_THRESHOLD);

    validate_arguments(
        "rgb_slice",
        rgb_slice.len(),
        RGB_COMPONENTS,
        k,
        max_iter,
        convergence_threshold,
    )?;

    Ok(quantize_packed_colors(
        rgb_slice,
        RGB_COMPONENTS,
        k,
        max_iter,
        convergence_threshold,
        seed,
    ))
}

/// Map packed pixels onto a palette, for the second half of color quantization.
#[wasm_bindgen(skip_typescript)]
pub fn apply_palette(pixels: &[u8], palette: &[u8], components: usize) -> Result<Vec<u8>, JsValue> {
    validate_palette_inputs(pixels.len(), palette.len(), components)?;
    Ok(packed_histogram::map_to_rgba(pixels, palette, components))
}

/// Map packed pixels onto their nearest palette entry indices.
#[wasm_bindgen(skip_typescript)]
pub fn apply_palette_indices(
    pixels: &[u8],
    palette: &[u8],
    components: usize,
) -> Result<Vec<u32>, JsValue> {
    validate_palette_inputs(pixels.len(), palette.len(), components)?;
    Ok(packed_histogram::map_to_indices(
        pixels, palette, components,
    ))
}

/// Map packed pixels onto a palette using Floyd–Steinberg error-diffusion dithering.
#[wasm_bindgen(skip_typescript)]
pub fn apply_palette_dithered(
    pixels: &[u8],
    width: usize,
    height: usize,
    palette: &[u8],
    components: usize,
) -> Result<Vec<u8>, JsValue> {
    validate_palette_inputs(pixels.len(), palette.len(), components)?;
    let expected_len = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(components));
    if expected_len != Some(pixels.len()) {
        return Err(range_error(
            "Error: pixels length must equal width * height * components.",
        ));
    }
    Ok(packed_histogram::map_to_rgba_dithered(
        pixels, width, height, palette, components,
    ))
}

#[wasm_bindgen(skip_typescript)]
/// Find the k-means centroids of an RGBA u8 slice for color quantization.
pub fn kmeans_rgba(
    rgba_slice: &[u8],
    k: usize,
    max_iter: usize,
    convergence_threshold: Option<f64>,
    seed: Option<u32>,
) -> Result<Vec<u8>, JsValue> {
    let convergence_threshold = convergence_threshold.unwrap_or(PACKED_CONVERGENCE_THRESHOLD);

    validate_arguments(
        "rgba_slice",
        rgba_slice.len(),
        RGBA_COMPONENTS,
        k,
        max_iter,
        convergence_threshold,
    )?;

    Ok(quantize_packed_colors(
        rgba_slice,
        RGBA_COMPONENTS,
        k,
        max_iter,
        convergence_threshold,
        seed,
    ))
}

#[wasm_bindgen(skip_typescript)]
/// Find the e-means centroids of an RGBA u8 slice for color quantization.
pub fn emeans_rgba(
    rgba_slice: &[u8],
    k: usize,
    epochs: Option<u32>,
    batch_size: Option<u32>,
    seed: Option<u32>,
) -> Result<Vec<u8>, JsValue> {
    let epochs = epochs.map_or(emeans::DEFAULT_EPOCHS, |value| value as usize);
    let batch_size = batch_size.map(|value| value as usize);

    validate_emeans_arguments(
        "rgba_slice",
        rgba_slice.len(),
        RGBA_COMPONENTS,
        k,
        epochs,
        batch_size,
    )?;

    Ok(quantize_packed_colors_emeans(
        rgba_slice,
        RGBA_COMPONENTS,
        k,
        epochs,
        batch_size,
        seed,
    ))
}

#[wasm_bindgen(typescript_custom_section)]
const KMEANS_TYPE: &'static str = r#"
export type ByteBuffer = Uint8Array | Uint8ClampedArray;

export interface IKmeansResult {
    /** The number of iterations performed until convergence, or completed passes for `emeans` */
    it: number;
    /** The cluster size */
    k: number;
    /** The value for each centroid of the cluster */
    centroids: number[][];
    /** The index to the centroid corresponding to each value of the data array */
    idxs: Uint32Array;
    /** Function to test new point membership */
    test: (point: number[], fnDist?: (a: number[], b: number[]) => number) => number;
}

export interface IFlatKmeansResultF64 {
    /** The number of iterations performed until convergence, or completed passes for `emeans_f64` */
    it: number;
    /** The cluster size */
    k: number;
    /** Number of components per vector */
    dimensions: number;
    /** Flat row-major centroids (`k * dimensions` values) */
    centroids: Float64Array;
    /** The index of the centroid corresponding to each input point */
    idxs: Uint32Array;
}

export interface IFlatKmeansResultF32 {
    /** The number of iterations performed until convergence, or completed passes for `emeans_f32` */
    it: number;
    /** The cluster size */
    k: number;
    /** Number of components per vector */
    dimensions: number;
    /** Flat row-major centroids (`k * dimensions` values) */
    centroids: Float32Array;
    /** The index of the centroid corresponding to each input point */
    idxs: Uint32Array;
}

/**
 * Find the k-means centroids for any vector-space.
 */
export function kmeans(
    data: Array<Array<number>>,
    k: number,
    max_iter: number,
    convergence_threshold?: number | null,
    seed?: number | null,
): IKmeansResult;

/**
 * Find the k-means centroids for a flat `Float64Array` of row-major vectors.
 */
export function kmeans_f64(
    data: Float64Array,
    dimensions: number,
    k: number,
    max_iter: number,
    convergence_threshold?: number | null,
    seed?: number | null,
): IFlatKmeansResultF64;

/**
 * Find the k-means centroids for a flat `Float32Array` of row-major vectors.
 */
export function kmeans_f32(
    data: Float32Array,
    dimensions: number,
    k: number,
    max_iter: number,
    convergence_threshold?: number | null,
    seed?: number | null,
): IFlatKmeansResultF32;

/**
 * Find the e-means centroids for any vector-space.
 */
export function emeans(
    data: Array<Array<number>>,
    k: number,
    epochs?: number | null,
    batch_size?: number | null,
    seed?: number | null,
): IKmeansResult;

/**
 * Find the e-means centroids for a flat `Float64Array` of row-major vectors.
 */
export function emeans_f64(
    data: Float64Array,
    dimensions: number,
    k: number,
    epochs?: number | null,
    batch_size?: number | null,
    seed?: number | null,
): IFlatKmeansResultF64;

/**
 * Find the e-means centroids for a flat `Float32Array` of row-major vectors.
 */
export function emeans_f32(
    data: Float32Array,
    dimensions: number,
    k: number,
    epochs?: number | null,
    batch_size?: number | null,
    seed?: number | null,
): IFlatKmeansResultF32;

/**
 * Find the k-means centroids of an RGB buffer (`Uint8Array` or `Uint8ClampedArray`).
 */
export function kmeans_rgb(
    rgb_slice: ByteBuffer,
    k: number,
    max_iter: number,
    convergence_threshold?: number | null,
    seed?: number | null,
): Uint8Array;

/**
 * Find the k-means centroids of an RGBA buffer (`Uint8Array` or `Uint8ClampedArray`).
 */
export function kmeans_rgba(
    rgba_slice: ByteBuffer,
    k: number,
    max_iter: number,
    convergence_threshold?: number | null,
    seed?: number | null,
): Uint8Array;

/**
 * Find the e-means centroids of an RGB buffer (`Uint8Array` or `Uint8ClampedArray`).
 */
export function emeans_rgb(
    rgb_slice: ByteBuffer,
    k: number,
    epochs?: number | null,
    batch_size?: number | null,
    seed?: number | null,
): Uint8Array;

/**
 * Find the e-means centroids of an RGBA buffer (`Uint8Array` or `Uint8ClampedArray`).
 */
export function emeans_rgba(
    rgba_slice: ByteBuffer,
    k: number,
    epochs?: number | null,
    batch_size?: number | null,
    seed?: number | null,
): Uint8Array;

/**
 * Map packed pixels onto a palette, returning RGBA bytes ready for `ImageData`.
 */
export function apply_palette(
    pixels: ByteBuffer,
    palette: ByteBuffer,
    components: number,
): Uint8Array;

/**
 * Map packed pixels onto their nearest palette entry indices (`0..palette.length / components`).
 */
export function apply_palette_indices(
    pixels: ByteBuffer,
    palette: ByteBuffer,
    components: number,
): Uint32Array;

/**
 * Map packed pixels onto a palette using Floyd-Steinberg error-diffusion dithering.
 */
export function apply_palette_dithered(
    pixels: ByteBuffer,
    width: number,
    height: number,
    palette: ByteBuffer,
    components: number,
): Uint8Array;
"#;

#[wasm_bindgen(skip_typescript)]
/// Find the k-means centroids for any vector-space.
pub fn kmeans(
    data: Array,
    k: usize,
    max_iter: usize,
    convergence_threshold: Option<f64>,
    seed: Option<u32>,
) -> Result<Object, JsValue> {
    let convergence_threshold = convergence_threshold.unwrap_or(0.0);
    validate_kmeans_common(k, max_iter, convergence_threshold)?;

    let (points, dimension, _point_count) = flatten_vector_data(&data)?;

    let result = kmeans_triangle::hamerly_kmeans_dispatched_weighted(
        k,
        max_iter,
        convergence_threshold,
        &points,
        dimension,
        kmeans_triangle::Optional {
            seed,
            ..Default::default()
        },
    );

    build_result_object(
        k,
        result.iterations,
        &result.centroids,
        result.dimensions,
        &result.assignments(),
    )
}

#[wasm_bindgen(skip_typescript)]
/// Find the k-means centroids for a flat `Float64Array` of row-major vectors.
pub fn kmeans_f64(
    data: &[f64],
    dimensions: usize,
    k: usize,
    max_iter: usize,
    convergence_threshold: Option<f64>,
    seed: Option<u32>,
) -> Result<Object, JsValue> {
    let convergence_threshold = convergence_threshold.unwrap_or(0.0);
    validate_kmeans_common(k, max_iter, convergence_threshold)?;
    validate_flat_buffer_shape(data.len(), dimensions)?;
    if data.iter().any(|value| !value.is_finite()) {
        return Err(range_error(
            "Error: Data points must contain only finite numbers.",
        ));
    }

    let result = kmeans_triangle::hamerly_kmeans_dispatched_weighted(
        k,
        max_iter,
        convergence_threshold,
        data,
        dimensions,
        kmeans_triangle::Optional {
            seed,
            ..Default::default()
        },
    );

    build_flat_result_f64(
        k,
        result.iterations,
        result.dimensions,
        &result.centroids,
        &result.assignments(),
    )
}

#[wasm_bindgen(skip_typescript)]
/// Find the k-means centroids for a flat `Float32Array` of row-major vectors.
pub fn kmeans_f32(
    data: &[f32],
    dimensions: usize,
    k: usize,
    max_iter: usize,
    convergence_threshold: Option<f64>,
    seed: Option<u32>,
) -> Result<Object, JsValue> {
    let convergence_threshold = convergence_threshold.unwrap_or(0.0);
    validate_kmeans_common(k, max_iter, convergence_threshold)?;
    validate_flat_buffer_shape(data.len(), dimensions)?;

    let mut points = Vec::with_capacity(data.len());
    for value in data {
        if !value.is_finite() {
            return Err(range_error(
                "Error: Data points must contain only finite numbers.",
            ));
        }
        points.push(f64::from(*value));
    }

    let result = kmeans_triangle::hamerly_kmeans_dispatched_weighted(
        k,
        max_iter,
        convergence_threshold,
        &points,
        dimensions,
        kmeans_triangle::Optional {
            seed,
            ..Default::default()
        },
    );

    build_flat_result_f32(
        k,
        result.iterations,
        result.dimensions,
        &result.centroids,
        &result.assignments(),
    )
}

fn flatten_vector_data(data: &Array) -> Result<(Vec<f64>, usize, usize), JsValue> {
    let point_count = data.length() as usize;
    if point_count == 0 {
        return Ok((Vec::new(), 0, 0));
    }

    let first_point: Array = data
        .get(0)
        .dyn_into()
        .map_err(|_| type_error("Error: Each data point must be an array of finite numbers."))?;
    let dimension = first_point.length() as usize;
    let mut points = Vec::with_capacity(point_count * dimension);

    for row_index in 0..point_count {
        let point_array: Array = data.get(row_index as u32).dyn_into().map_err(|_| {
            type_error("Error: Each data point must be an array of finite numbers.")
        })?;
        if point_array.length() as usize != dimension {
            return Err(range_error(
                "Error: All data points must have the same dimension.",
            ));
        }
        for col_index in 0..dimension {
            let value = point_array.get(col_index as u32).as_f64().ok_or_else(|| {
                type_error("Error: Each data point must be an array of finite numbers.")
            })?;
            if !value.is_finite() {
                return Err(range_error(
                    "Error: Data points must contain only finite numbers.",
                ));
            }
            points.push(value);
        }
    }

    Ok((points, dimension, point_count))
}

/// Builds the JavaScript result shape shared by `kmeans` and `emeans`.
fn build_result_object(
    k: usize,
    iterations: usize,
    centroids: &[f64],
    dimensions: usize,
    assignments: &[u32],
) -> Result<Object, JsValue> {
    // `Array::new_with_length` would pre-fill the array with holes, so every
    // pushed centroid would land after `k` empty slots.
    let centroids_array = Array::new();
    if dimensions > 0 {
        for centroid in centroids.chunks_exact(dimensions) {
            let centroid_array = Array::new();
            for value in centroid {
                centroid_array.push(&JsValue::from_f64(*value));
            }
            centroids_array.push(&centroid_array);
        }
    }

    let p_c = Uint32Array::from(assignments);

    let result_js = Object::new();
    Reflect::set(
        &result_js,
        &JsValue::from_str("k"),
        &JsValue::from_f64(k as f64),
    )?;
    Reflect::set(
        &result_js,
        &JsValue::from_str("it"),
        &JsValue::from_f64(iterations as f64),
    )?;
    Reflect::set(
        &result_js,
        &JsValue::from_str("centroids"),
        &centroids_array,
    )?;
    Reflect::set(&result_js, &JsValue::from_str("idxs"), &p_c)?;
    Reflect::set(&result_js, &JsValue::from_str("test"), &make_test_fn())?;

    Ok(result_js)
}

fn build_flat_result(
    k: usize,
    iterations: usize,
    dimensions: usize,
    centroids: &JsValue,
    assignments: &[u32],
) -> Result<Object, JsValue> {
    let result_js = Object::new();
    Reflect::set(
        &result_js,
        &JsValue::from_str("k"),
        &JsValue::from_f64(k as f64),
    )?;
    Reflect::set(
        &result_js,
        &JsValue::from_str("it"),
        &JsValue::from_f64(iterations as f64),
    )?;
    Reflect::set(
        &result_js,
        &JsValue::from_str("dimensions"),
        &JsValue::from_f64(dimensions as f64),
    )?;
    Reflect::set(&result_js, &JsValue::from_str("centroids"), centroids)?;
    Reflect::set(
        &result_js,
        &JsValue::from_str("idxs"),
        &Uint32Array::from(assignments),
    )?;
    Ok(result_js)
}

fn build_flat_result_f64(
    k: usize,
    iterations: usize,
    dimensions: usize,
    centroids: &[f64],
    assignments: &[u32],
) -> Result<Object, JsValue> {
    let centroids_js: JsValue = Float64Array::from(centroids).into();
    build_flat_result(k, iterations, dimensions, &centroids_js, assignments)
}

fn build_flat_result_f32(
    k: usize,
    iterations: usize,
    dimensions: usize,
    centroids: &[f64],
    assignments: &[u32],
) -> Result<Object, JsValue> {
    let centroids_f32: Vec<f32> = centroids.iter().map(|value| *value as f32).collect();
    let centroids_js: JsValue = Float32Array::from(centroids_f32.as_slice()).into();
    build_flat_result(k, iterations, dimensions, &centroids_js, assignments)
}

#[wasm_bindgen(skip_typescript)]
/// Find the e-means centroids for any vector-space.
pub fn emeans(
    data: Array,
    k: usize,
    epochs: Option<u32>,
    batch_size: Option<u32>,
    seed: Option<u32>,
) -> Result<Object, JsValue> {
    let epochs = epochs.map_or(emeans::DEFAULT_EPOCHS, |value| value as usize);
    let batch_size = batch_size.map(|value| value as usize);
    validate_emeans_common(k, epochs, batch_size)?;

    let (points, dimension, _point_count) = flatten_vector_data(&data)?;

    let result = emeans::emeans_dispatched(
        k,
        epochs,
        &points,
        dimension,
        emeans::EmeansOptions {
            batch_size,
            capture_assignments: true,
            seed,
            ..Default::default()
        },
    );

    build_result_object(
        k,
        result.epochs,
        &result.centroids,
        result.dimensions,
        result.assignments(),
    )
}

#[wasm_bindgen(skip_typescript)]
/// Find the e-means centroids for a flat `Float64Array` of row-major vectors.
pub fn emeans_f64(
    data: &[f64],
    dimensions: usize,
    k: usize,
    epochs: Option<u32>,
    batch_size: Option<u32>,
    seed: Option<u32>,
) -> Result<Object, JsValue> {
    let epochs = epochs.map_or(emeans::DEFAULT_EPOCHS, |value| value as usize);
    let batch_size = batch_size.map(|value| value as usize);
    validate_emeans_common(k, epochs, batch_size)?;
    validate_flat_buffer_shape(data.len(), dimensions)?;
    if data.iter().any(|value| !value.is_finite()) {
        return Err(range_error(
            "Error: Data points must contain only finite numbers.",
        ));
    }

    let result = emeans::emeans_dispatched(
        k,
        epochs,
        data,
        dimensions,
        emeans::EmeansOptions {
            batch_size,
            capture_assignments: true,
            seed,
            ..Default::default()
        },
    );

    build_flat_result_f64(
        k,
        result.epochs,
        result.dimensions,
        &result.centroids,
        result.assignments(),
    )
}

#[wasm_bindgen(skip_typescript)]
/// Find the e-means centroids for a flat `Float32Array` of row-major vectors.
pub fn emeans_f32(
    data: &[f32],
    dimensions: usize,
    k: usize,
    epochs: Option<u32>,
    batch_size: Option<u32>,
    seed: Option<u32>,
) -> Result<Object, JsValue> {
    let epochs = epochs.map_or(emeans::DEFAULT_EPOCHS, |value| value as usize);
    let batch_size = batch_size.map(|value| value as usize);
    validate_emeans_common(k, epochs, batch_size)?;
    validate_flat_buffer_shape(data.len(), dimensions)?;

    let mut use_f32_assignment = true;
    let mut points = Vec::with_capacity(data.len());
    for value in data {
        if !value.is_finite() {
            return Err(range_error(
                "Error: Data points must contain only finite numbers.",
            ));
        }
        if value.abs() >= 1e15 {
            use_f32_assignment = false;
        }
        points.push(f64::from(*value));
    }

    let result = emeans::emeans_dispatched(
        k,
        epochs,
        &points,
        dimensions,
        emeans::EmeansOptions {
            batch_size,
            capture_assignments: true,
            seed,
            use_f32_assignment,
            points_f32: use_f32_assignment.then_some(data),
            ..Default::default()
        },
    );

    build_flat_result_f32(
        k,
        result.epochs,
        result.dimensions,
        &result.centroids,
        result.assignments(),
    )
}
