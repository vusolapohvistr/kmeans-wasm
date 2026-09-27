// Degenerate shapes: k larger than the number of distinct values, k=2 on a single
// colour, and a palette that should be all zeros rather than all black-by-NaN.
use kmeans_wasm::kmeans_rgb;

fn rgb(values: &[u8]) -> Vec<u8> {
    values.to_vec()
}

#[test]
fn k_larger_than_the_distinct_count() {
    for (pixels, distinct, k) in [
        (vec![10u8, 20, 30], 1usize, 8usize),
        (vec![10, 20, 30, 40, 50, 60], 2, 8),
        (vec![10, 20, 30, 10, 20, 30, 40, 50, 60], 3, 16),
        (vec![0u8, 0, 0, 255, 255, 255], 2, 32),
    ] {
        let palette = kmeans_rgb(rgb(&pixels), k, 100, Some(0.1)).unwrap();
        assert_eq!(palette.len(), k * 3, "k={k} distinct={distinct}");
        // Every centroid must be a real u8, which it is by construction, and the
        // palette must be drawn from the input's own colours.
        let (centroids, _) = palette.as_chunks::<3>();
        let (colours, _) = pixels.as_chunks::<3>();
        for centroid in centroids {
            assert!(
                colours.contains(centroid),
                "centroid {centroid:?} is not one of the input colours"
            );
        }
    }
}

#[test]
fn every_centroid_is_reachable_from_some_pixel() {
    // With coincident centroids some entry can attract nothing, and a palette
    // entry no pixel maps to is a colour the quantized image can never show.
    let mut pixels = Vec::new();
    for value in 0..8u8 {
        pixels.extend_from_slice(&[value * 30, 255 - value * 30, value * 17]);
    }
    let seed: Vec<u8> = pixels.clone();
    for _ in 0..500 {
        pixels.extend_from_slice(&seed);
    }
    let (rows, _) = pixels.as_chunks::<3>();
    let distinct: std::collections::HashSet<[u8; 3]> =
        rows.iter().map(|row| [row[0], row[1], row[2]]).collect();
    let distinct = distinct.len();
    for k in [distinct + 1, distinct * 2, 64] {
        let palette = kmeans_rgb(pixels.clone(), k, 100, Some(0.1)).unwrap();
        let (centroids, _) = palette.as_chunks::<3>();
        let reachable = centroids.iter().filter(|c| rows.contains(c)).count();
        assert!(
            reachable >= distinct.min(k),
            "k={k} distinct={distinct}: only {reachable} of {} centroids are a real colour",
            palette.len() / 3
        );
    }
}

#[test]
fn a_single_colour_input_gives_that_colour() {
    let palette = kmeans_rgb(vec![10, 20, 30], 2, 100, Some(0.1)).unwrap();
    assert_eq!(palette.len(), 6);
    let (centroids, _) = palette.as_chunks::<3>();
    for centroid in centroids {
        assert_eq!(
            *centroid,
            [10, 20, 30],
            "the only colour available is the only answer"
        );
    }
}
