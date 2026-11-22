pub mod emeans;
pub mod kmeans_triangle;
pub mod packed_histogram;
#[cfg(feature = "counters")]
pub mod work;

use js_sys::{Array, Function, Object, Reflect};
use wasm_bindgen::{JsCast, prelude::*};

/// Number of vector components clustered by [`kmeans_rgb`].
const RGB_COMPONENTS: usize = 3;
/// Number of vector components clustered by [`kmeans_rgba`].
const RGBA_COMPONENTS: usize = 4;

fn validate_arguments(
    slice_name: &str,
    slice_len: usize,
    components: usize,
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
) -> Result<(), JsValue> {
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

    if convergence_threshold.is_sign_negative() {
        return Err(JsValue::from_str(
            "Error: convergence_threshold must be positive",
        ));
    }

    if !slice_len.is_multiple_of(components) {
        return Err(JsValue::from_str(&format!(
            "Error: The length of {slice_name} must be a multiple of {components}."
        )));
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
    if k < 2 {
        return Err(JsValue::from_str(
            "Error: k must be greater than or equal to 2.",
        ));
    }

    if epochs < 1 {
        return Err(JsValue::from_str(
            "Error: epochs must be greater than or equal to 1.",
        ));
    }

    if batch_size.is_some_and(|size| size < 1) {
        return Err(JsValue::from_str(
            "Error: batch_size must be greater than or equal to 1.",
        ));
    }

    if !slice_len.is_multiple_of(components) {
        return Err(JsValue::from_str(&format!(
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
        return Err(JsValue::from_str(
            "Error: k must be greater than or equal to 2.",
        ));
    }

    if epochs < 1 {
        return Err(JsValue::from_str(
            "Error: epochs must be greater than or equal to 1.",
        ));
    }

    if batch_size.is_some_and(|size| size < 1) {
        return Err(JsValue::from_str(
            "Error: batch_size must be greater than or equal to 1.",
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
///
/// Measured against running to `max_iter`: on flat-region imagery, which is
/// most canvas, PNG, and UI input, the palette comes back byte-identical after
/// two rounds instead of a hundred. On inputs where every value is distinct the
/// palette either matches exactly or moves by one 8-bit level. Anything larger
/// is unsafe here, because the threshold scales with `k` while the output does
/// not: a `k / 4` threshold stopped before doing any work on a flat image and
/// shifted the palette by 254 levels.
const PACKED_CONVERGENCE_THRESHOLD: f64 = 0.1;

fn quantize_packed_colors(
    slice: Vec<u8>,
    components: usize,
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
) -> Vec<u8> {
    // Packed colour input repeats heavily, and equal pixels are interchangeable:
    // they take the same cluster and add the same amount to that cluster's sum.
    // Clustering the distinct values once, weighted by how often each occurs,
    // therefore returns the same centroids, and it does so on a point set that
    // can be orders of magnitude smaller.
    //
    // This is exact rather than approximate, and specifically so here. The core
    // sums `w * v` once per distinct colour here and `1 * v` once per pixel on
    // the full path, so the two accumulate in different orders. That cannot
    // change the result for `u8` input: every value and every partial sum is a
    // small integer, far below the 2^53 where `f64` stops representing integers
    // exactly, and integer addition is associative. Verified bit for bit from 20
    // random starting points.
    //
    // `collapse` returns `None` when the input holds more distinct values than
    // its table can hold, which is the case where the reduction would be worth
    // less than about a factor of five anyway. That path costs one bounded pass
    // over the input and measured 0.97x to 1.03x, inside the noise of this
    // machine.
    let collapsed = packed_histogram::collapse(&slice, components);
    let widened;
    let (points, weights): (&[f64], Option<&[f64]>) = match collapsed.as_ref() {
        Some(found) => (found.points.as_slice(), Some(found.weights.as_slice())),
        None => {
            // One flat f64 buffer for the whole input, so the clustering core
            // reads contiguous memory instead of chasing a vector per point. The
            // explicit capacity lets this widening loop vectorize instead of
            // growing per element.
            let mut buffer = Vec::with_capacity(slice.len());
            for value in slice.iter() {
                buffer.push(*value as f64);
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
            ..Default::default()
        },
    );

    centroids
        .centroids
        .iter()
        .map(|value| *value as u8)
        .collect()
}

fn quantize_packed_colors_emeans(
    slice: Vec<u8>,
    components: usize,
    k: usize,
    epochs: usize,
    batch_size: Option<usize>,
) -> Vec<u8> {
    // Same reduction as the Lloyd path. For e-means the weight is also the
    // mechanism that keeps the batch statistics faithful: a distinct colour
    // contributes its whole pixel count to the batch that contains it.
    let collapsed = packed_histogram::collapse(&slice, components);
    let widened;
    let (points, weights): (&[f64], Option<&[f64]>) = match collapsed.as_ref() {
        Some(found) => (found.points.as_slice(), Some(found.weights.as_slice())),
        None => {
            let mut buffer = Vec::with_capacity(slice.len());
            for value in slice.iter() {
                buffer.push(*value as f64);
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
            ..Default::default()
        },
    );

    result.centroids.iter().map(|value| *value as u8).collect()
}

#[wasm_bindgen]
/// Find the e-means centroids of an RGB u8 slice for color quantization.
///
/// e-means is the mini-batch EMA method of Bessa et al., *e-means: outperforming
/// k-means (in one-tenth the epochs)*, arXiv:2611.09501. It is an alternative to
/// [`kmeans_rgb`], not a replacement: the Lloyd path is deterministic for a
/// given draw and the e-means path is a stochastic mini-batch method.
///
/// - `rgb_slice` - Uint8Array of RGB components, where each component is a u8 value.
/// - `k >= 2` - number of clusters.
/// - `epochs` - passes over the input. Defaults to 5, the reference
///   implementation's default.
/// - `batch_size` - points per EMA update. Defaults to the paper's scaling law,
///   which depends on `k`, the input size, `epochs` and the component count.
///
/// The result is a palette of `k` RGB centroids, using last-epoch-mean
/// finalization. Unlike [`kmeans_rgb`] there is no convergence threshold: the
/// cosine schedule fixes the number of updates.
pub fn emeans_rgb(
    rgb_slice: Vec<u8>,
    k: usize,
    epochs: Option<u32>,
    batch_size: Option<u32>,
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
    ))
}

#[wasm_bindgen]
/// Find the k-means centroids of an RGB u8 slice for color quantization.
///
/// - `rgb_slice` - Uint8Array of RGB components, where each component is a u8 value.
/// - `k >= 2` - number of clusters.
/// - `max_iter >= 1` - maximum number of iterations.
/// - `convergence_threshold > 0.0` - the threshold to determine when the centroids have
///   converged. Defaults to 0.1, which is below the resolution of the `u8` result. Pass 0 to
///   always run the full `max_iter`.
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
    ))
}

/// Map packed pixels onto a palette, for the second half of color quantization.
///
/// Clustering produces the palette; this produces the quantized image. Callers
/// otherwise write this loop themselves, and a full nearest-palette search per
/// pixel is `n * k`, so a megapixel at 32 colours is 29 million distance
/// evaluations in JavaScript.
///
/// This resolves each distinct colour to its nearest entry once and then maps
/// every pixel with a single table probe, so the cost is `distinct * k`
/// evaluations plus one probe per pixel. Real images repeat heavily, so that is
/// usually an order of magnitude less work. Measured through wasm on the four
/// photographs in `js_bench/images` against the JavaScript loop this replaces,
/// all at k=32: 2.00x on the 6.2% distinct one, 1.59x at 30.5%, 1.24x at 36.6% and
/// 1.56x at 45.8%, with byte-identical output every time.
///
/// - `pixels` - Uint8Array of packed pixels, `components` values each.
/// - `palette` - the centroids returned by `kmeans_rgb` or `kmeans_rgba`, so with
///   the same component count.
/// - `components` - 3 or 4, matching both arguments.
///
/// Returns RGBA, always four values per pixel, which is what `putImageData`
/// takes. A three-component input gets a fully opaque alpha. The result is
/// byte-identical to searching the palette for every pixel in JavaScript with a
/// strict `<` comparison, so a tie resolves to the lower index.
#[wasm_bindgen]
pub fn apply_palette(
    pixels: Vec<u8>,
    palette: Vec<u8>,
    components: usize,
) -> Result<Vec<u8>, JsValue> {
    if !matches!(components, RGB_COMPONENTS | RGBA_COMPONENTS) {
        return Err(JsValue::from_str(
            "Error: components must be 3 for RGB or 4 for RGBA.",
        ));
    }
    if !pixels.len().is_multiple_of(components) {
        return Err(JsValue::from_str(&format!(
            "Error: The length of pixels must be a multiple of {components}."
        )));
    }
    if palette.is_empty() || !palette.len().is_multiple_of(components) {
        return Err(JsValue::from_str(
            "Error: The length of palette must be a non-zero multiple of components.",
        ));
    }

    Ok(map_to_rgba(&pixels, &palette, components))
}

/// The mapping, as an RGBA buffer ready for `putImageData`.
fn map_to_rgba(pixels: &[u8], palette: &[u8], components: usize) -> Vec<u8> {
    let pixel_count = pixels.len() / components;
    let mut output = vec![255u8; pixel_count * RGBA_COMPONENTS];

    if pixel_count == 0 {
        return output;
    }

    // The table needs one slot per distinct colour with room to probe, so it is
    // sized from the pixel count and gives up if the input does not compress.
    let map = packed_histogram::PaletteMap::build(pixels, components, palette);

    let source_is_rgba = components == RGBA_COMPONENTS;
    let (rows, _) = output.as_chunks_mut::<RGBA_COMPONENTS>();
    for (pixel, mapped) in pixels.chunks_exact(components).zip(rows.iter_mut()) {
        let entry = match map {
            Some(ref table) => table.lookup(pixel),
            // No table, so pay for a search. Only on input with almost no repeated
            // colours, and bounded at about 0.83x by the build that was attempted.
            None => packed_histogram::nearest_palette(pixel, palette, components),
        };
        let base = entry * components;
        mapped[0] = palette[base];
        mapped[1] = palette[base + 1];
        mapped[2] = palette[base + 2];
        mapped[3] = if source_is_rgba {
            palette[base + 3]
        } else {
            255
        };
    }

    output
}

#[wasm_bindgen]
/// Find the k-means centroids of an RGBA u8 slice for color quantization.
///
/// - `rgba_slice` - Uint8Array of RGBA components, where each component is a u8 value.
/// - `k >= 2` - number of clusters.
/// - `max_iter >= 1` - maximum number of iterations.
/// - `convergence_threshold > 0.0` - the threshold to determine when the centroids have
///   converged. Defaults to 0.1, which is below the resolution of the `u8` result. Pass 0 to
///   always run the full `max_iter`.
///
/// This function is the four-component counterpart of `kmeans_rgb`. It is the drop-in choice for
/// `ImageData.data` and other buffers that interleave red, green, blue, and alpha, so no repacking
/// is needed before clustering. The alpha channel is clustered like any other component, which makes
/// the function useful for image data that mixes transparent and opaque pixels. The resulting
/// centroids represent the quantized RGBA colors.
pub fn kmeans_rgba(
    rgba_slice: Vec<u8>,
    k: usize,
    max_iter: usize,
    convergence_threshold: Option<f64>,
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
    ))
}

#[wasm_bindgen]
/// Find the e-means centroids of an RGBA u8 slice for color quantization.
///
/// This is the four-component counterpart of [`emeans_rgb`], and the e-means
/// alternative to [`kmeans_rgba`]. The alpha channel is clustered like any other
/// component.
///
/// - `rgba_slice` - Uint8Array of RGBA components, where each component is a u8 value.
/// - `k >= 2` - number of clusters.
/// - `epochs` - passes over the input. Defaults to 5.
/// - `batch_size` - points per EMA update. Defaults to the paper's scaling law.
pub fn emeans_rgba(
    rgba_slice: Vec<u8>,
    k: usize,
    epochs: Option<u32>,
    batch_size: Option<u32>,
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
    ))
}

#[wasm_bindgen(typescript_custom_section)]
const KMEANS_TYPE: &'static str = r#"
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

/**
* Find the e-means centroids for any vector-space.
*
* e-means is a mini-batch EMA method from Bessa et al., arXiv:2611.09501. It
* uses a fixed number of passes instead of a convergence threshold, and its
* batch size and initial step come from the paper's scaling laws. `it` is the
* number of completed passes. `idxs` records each point's assignment from the
* final pass, which is not guaranteed to be the nearest final centroid because
* the centroids move during that pass; use `test` for a nearest-centroid query.
*
* @param {Array<Array<number>>} data
* @param {number} k
* @param {number?} epochs
* @param {number?} batch_size
* @returns {IKmeansResult}
*/
export function emeans(data: Array<Array<number>>, k: number, epochs?: number, batch_size?: number): IKmeansResult;
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

    let (points, dimension, _point_count) = flatten_vector_data(&data)?;

    let result = kmeans_triangle::hamerly_kmeans_dispatched(
        k,
        max_iter,
        convergence_threshold,
        &points,
        dimension,
    );

    build_result_object(
        k,
        result.iterations,
        &result.centroids,
        result.dimensions,
        &result.assignments(),
    )
}

fn flatten_vector_data(data: &Array) -> Result<(Vec<f64>, usize, usize), JsValue> {
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

    let dimension = data_vec.first().map_or(0, |first_point| first_point.len());
    let point_count = data_vec.len();

    let mut points: Vec<f64> = Vec::with_capacity(point_count * dimension);
    for point in &data_vec {
        points.extend_from_slice(point);
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

    let p_c = js_sys::Uint32Array::new_with_length(assignments.len() as u32);
    for (index, point_index) in assignments.iter().enumerate() {
        p_c.set_index(index as u32, *point_index);
    }

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
    Reflect::set(
        &result_js,
        &JsValue::from_str("test"),
        &Function::new_with_args(
            "point, fnDist",
            "
        const centroids = this.centroids;
        if (centroids.length === 0 || point.length !== centroids[0].length) {
            throw new Error('Point should have the same length as centroid');
        }

        let minCentroid = 0;
        let minDist = Number.MAX_VALUE;
        const dist = fnDist ?? ((a, b) => {
            let result = 0;
            for (let i = 0; i < a.length; i++) {
                result += (a[i] - b[i]) ** 2;
            }

            return Math.sqrt(result);
        });

        for (const [i, centroid] of centroids.entries()) {
            const centroidDist = dist(centroid, point);
            if (centroidDist < minDist) {
                minDist = centroidDist;
                minCentroid = i;
            }
        }

        return minCentroid;
    ",
        ),
    )?;

    Ok(result_js)
}

#[wasm_bindgen(skip_typescript)]
/// Find the e-means centroids for any vector-space.
///
/// e-means is the mini-batch EMA method of Bessa et al., *e-means: outperforming
/// k-means (in one-tenth the epochs)*, arXiv:2611.09501. It is an alternative to
/// [`kmeans`] with a fixed pass budget and automatic batch and step sizes.
///
/// - `data` - array of arrays of number values, all of the same dimension.
/// - `k >= 2` - number of clusters.
/// - `epochs` - passes over the input. Defaults to 5.
/// - `batch_size` - points per EMA update. Defaults to the paper's scaling law.
///
/// The returned `it` is the number of completed passes. `idxs` holds the
/// assignment from the final pass; the centroids move during that pass, so an
/// entry is not guaranteed to be the nearest of the returned final centroids.
/// Use the returned `test` function for a nearest-centroid query.
pub fn emeans(
    data: Array,
    k: usize,
    epochs: Option<u32>,
    batch_size: Option<u32>,
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
