//! Native tests for the e-means entry points and core.
//!
//! The general `emeans` JavaScript API cannot run here because it builds a
//! `JsValue`, but the packed paths and the core are native code. Native entropy
//! is a fixed seed, so repeated calls are comparable.

use kmeans_wasm::emeans::{EmeansOptions, emeans_dispatched};
use kmeans_wasm::{emeans_rgb, emeans_rgba};

fn colours(bytes: &[u8]) -> Vec<[u8; 3]> {
    bytes.as_chunks::<3>().0.to_vec()
}

fn squared_distance(left: &[u8; 3], right: &[u8; 3]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let difference = f64::from(*a) - f64::from(*b);
            difference * difference
        })
        .sum()
}

fn mean_squared_error(pixels: &[[u8; 3]], palette: &[[u8; 3]]) -> f64 {
    let total: f64 = pixels
        .iter()
        .map(|pixel| {
            palette
                .iter()
                .map(|entry| squared_distance(pixel, entry))
                .fold(f64::MAX, f64::min)
        })
        .sum();
    total / pixels.len() as f64
}

#[test]
fn quantizes_rgb_values() {
    let rgb = vec![
        255, 0, 0, //
        0, 255, 0, //
        0, 0, 255, //
        255, 255, 0, //
        0, 255, 255, //
        255, 0, 255,
    ];

    let palette = emeans_rgb(rgb, 3, Some(5), None).unwrap();

    assert_eq!(palette.len(), 9);
}

#[test]
fn quantizes_rgba_values() {
    let rgba = vec![
        255, 0, 0, 255, //
        0, 255, 0, 255, //
        0, 0, 255, 128, //
        255, 255, 255, 0,
    ];

    let palette = emeans_rgba(rgba, 4, Some(5), Some(2)).unwrap();

    assert_eq!(palette.len(), 16);
}

#[test]
fn clusters_the_alpha_channel() {
    let mut rgba = Vec::new();
    for index in 0..64 {
        rgba.extend_from_slice(&[index * 4, 128, 64, if index % 2 == 0 { 0 } else { 255 }]);
    }

    let palette = emeans_rgba(rgba, 2, Some(5), None).unwrap();

    let alphas = palette
        .as_chunks::<4>()
        .0
        .iter()
        .map(|centroid| centroid[3])
        .collect::<Vec<_>>();
    assert!(
        alphas.contains(&0) && alphas.contains(&255),
        "expected one opaque and one transparent centroid, got {alphas:?}"
    );
}

/// Native entropy is seeded, so the same call has to return the same bytes. This
/// is not the bit-for-bit guarantee the Lloyd path carries across builds; it is
/// the weaker property that a run is repeatable within one build and process.
#[test]
fn repeated_runs_are_identical_natively() {
    let mut rgb = Vec::new();
    for index in 0..2000 {
        rgb.extend_from_slice(&[
            (index as u8).wrapping_mul(37),
            (index as u8).wrapping_mul(11),
            (index as u8).wrapping_mul(97),
        ]);
    }

    let first = emeans_rgb(rgb.clone(), 8, Some(5), None).unwrap();
    let second = emeans_rgb(rgb, 8, Some(5), None).unwrap();

    assert_eq!(first, second);
}

#[test]
fn a_single_colour_image_returns_that_colour() {
    for fill in [0u8, 64, 128, 255] {
        let rgb = vec![fill; 3 * 4096];

        let palette = colours(&emeans_rgb(rgb, 16, Some(5), None).unwrap());

        assert_eq!(palette.len(), 16);
        assert!(
            palette.iter().all(|centroid| *centroid == [fill; 3]),
            "expected every centroid to stay at {fill}, got {palette:?}"
        );
    }
}

#[test]
fn an_empty_buffer_returns_an_empty_palette() {
    assert_eq!(
        emeans_rgb(Vec::new(), 2, Some(5), None).unwrap(),
        Vec::<u8>::new()
    );
    assert_eq!(
        emeans_rgba(Vec::new(), 2, Some(5), None).unwrap(),
        Vec::<u8>::new()
    );
}

/// On separated groups the palette should recover the groups' means. This is a
/// quality floor, not a fingerprint: e-means is stochastic and the exact bytes
/// are allowed to move with the schedule.
#[test]
fn finds_separated_groups() {
    let groups = [
        [10u8, 10, 10],
        [10, 200, 10],
        [10, 10, 200],
        [200, 10, 10],
        [200, 200, 10],
        [200, 10, 200],
        [10, 200, 200],
        [200, 200, 200],
    ];
    let mut rgb = Vec::new();
    for (group, colour) in groups.iter().enumerate() {
        for _ in 0..(100 + group * 37) {
            rgb.extend_from_slice(colour);
        }
    }
    let pixels = colours(&rgb);

    let palette = colours(&emeans_rgb(rgb, groups.len(), Some(5), None).unwrap());

    assert_eq!(palette.len(), groups.len());
    assert_eq!(
        mean_squared_error(&pixels, &palette),
        0.0,
        "the palette should contain each separated group, got {palette:?}"
    );
}

/// The core accepts a histogram and must not depend on whether equal points are
/// stored once or repeated. With one epoch and one batch this is exact because
/// every weight lands in the same update.
#[test]
fn weighted_rows_match_expanded_points_in_one_batch() {
    let rows = [
        [30.0, 40.0],  //
        [30.0, 220.0], //
        [220.0, 40.0], //
        [220.0, 220.0],
    ];
    let weights = [100.0, 50.0, 25.0, 200.0];
    let mut collapsed = Vec::new();
    for row in rows {
        collapsed.extend_from_slice(&row);
    }
    let mut expanded = Vec::new();
    for (row, weight) in rows.iter().zip(weights.iter()) {
        for _ in 0..*weight as usize {
            expanded.extend_from_slice(row);
        }
    }
    let start = [30.0, 40.0, 30.0, 220.0, 220.0, 40.0, 220.0, 220.0];

    let weighted = emeans_dispatched(
        4,
        1,
        &collapsed,
        2,
        EmeansOptions {
            weights: Some(&weights),
            initial_centroids: Some(&start),
            batch_size: Some(1000),
            revival_threshold: Some(0.0),
            capture_assignments: false,
        },
    );
    let plain = emeans_dispatched(
        4,
        1,
        &expanded,
        2,
        EmeansOptions {
            weights: None,
            initial_centroids: Some(&start),
            batch_size: Some(1000),
            revival_threshold: Some(0.0),
            capture_assignments: false,
        },
    );

    assert_eq!(weighted.centroids, plain.centroids);
}
