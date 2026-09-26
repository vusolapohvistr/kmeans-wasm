use kmeans_wasm::{kmeans_rgb, kmeans_rgba};

#[test]
fn quantizes_rgb_values() {
    let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];

    let centroids = kmeans_rgb(rgb, 3, 100, Some(0.001)).unwrap();

    assert_eq!(centroids.len(), 9);
}

#[test]
fn quantizes_rgba_values() {
    let rgba = vec![
        255, 0, 0, 255, //
        0, 255, 0, 255, //
        0, 0, 255, 128, //
        255, 255, 255, 0,
    ];

    let centroids = kmeans_rgba(rgba, 4, 100, Some(0.001)).unwrap();

    assert_eq!(centroids.len(), 16);
}

#[test]
fn clusters_the_alpha_channel() {
    let mut rgba = Vec::new();
    for index in 0..64 {
        rgba.extend_from_slice(&[index * 4, 128, 64, if index % 2 == 0 { 0 } else { 255 }]);
    }

    let centroids = kmeans_rgba(rgba, 2, 100, Some(0.001)).unwrap();

    let alphas = centroids
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

/// A solid-colour image has a single distinct value, so at most one cluster can
/// hold a point and the rest are empty. An empty cluster has no mean, and
/// dividing by its zero count used to produce a NaN centroid, which saturates to
/// 0 on the way out. Asking for more colours than the image contains therefore
/// returned a palette of black entries, which for a white or coloured image is
/// plainly wrong rather than merely redundant.
#[test]
fn a_solid_colour_image_returns_no_spurious_black_entries() {
    for fill in [0u8, 64, 128, 255] {
        let rgb = vec![fill; 3 * 4096];

        let centroids = kmeans_rgb(rgb, 16, 100, Some(0.001)).unwrap();

        assert_eq!(centroids.len(), 48);
        assert!(
            centroids.iter().all(|value| *value == fill),
            "expected every centroid to stay at {fill}, got {centroids:?}"
        );
    }
}

/// The same starvation reached through the reachable route rather than a
/// degenerate one: pixels that are exactly equal. Two seeded centroids landing
/// on the same value tie, ties resolve to the lowest index, and the loser is
/// left with no points at all. This input has more distinct values than
/// clusters, so nothing here is forced.
#[test]
fn repeated_pixel_values_return_no_spurious_black_entries() {
    let distinct = 40usize;
    let mut rgb = Vec::new();
    for index in 0..8192 {
        let pick = index % distinct;
        rgb.extend_from_slice(&[(pick * 6) as u8, (pick * 3) as u8, (255 - pick * 6) as u8]);
    }

    let centroids = kmeans_rgb(rgb, 16, 100, Some(0.001)).unwrap();

    let blacks = centroids
        .as_chunks::<3>()
        .0
        .iter()
        .filter(|centroid| **centroid == [0u8, 0, 0])
        .count();
    assert_eq!(
        blacks, 0,
        "no input pixel is black, so no centroid may be black, got {centroids:?}"
    );
}
