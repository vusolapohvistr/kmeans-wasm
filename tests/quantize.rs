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
