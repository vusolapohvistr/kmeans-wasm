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

/// The packed default is 0.1, below the resolution of the `u8` result. On
/// flat-region input the centroids reach a fixpoint almost immediately, so
/// running to `max_iter` instead wastes rounds without changing a single byte
/// of the palette.
#[test]
fn the_default_threshold_stops_flat_input_early() {
    let mut rgb = Vec::new();
    for index in 0..4096 {
        let band = (index / 512) % 8;
        rgb.extend_from_slice(&[
            (band * 32) as u8,
            (band * 16) as u8,
            255 - (band * 32) as u8,
        ]);
    }

    let early = kmeans_rgb(rgb.clone(), 8, 100, None).unwrap();
    let full = kmeans_rgb(rgb, 8, 100, Some(0.0)).unwrap();

    assert_eq!(
        early, full,
        "the default threshold must not change the palette for flat input"
    );
}

/// Squared distance between two colours, in the same units the core uses.
fn squared_distance(left: &[u8; 3], right: &[u8; 3]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let difference = f64::from(*a) - f64::from(*b);
            difference * difference
        })
        .sum()
}

/// The colour in `palette` that `pixel` maps to.
fn nearest(palette: &[[u8; 3]], pixel: &[u8; 3]) -> [u8; 3] {
    palette
        .iter()
        .map(|entry| (squared_distance(pixel, entry), *entry))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("palette is not empty")
        .1
}

fn colours(bytes: &[u8]) -> Vec<[u8; 3]> {
    bytes.as_chunks::<3>().0.to_vec()
}

/// Every palette entry must be reachable: at least one pixel maps to it. A dead
/// centroid wastes one of the caller's `k` slots and hands back a colour the
/// quantized image never uses.
///
/// The guarantee only holds at or below the number of distinct colours, which is
/// where this stays. Above it, empty clusters are forced rather than unlucky, and
/// a forced cluster legitimately has no pixels.
#[test]
fn every_palette_entry_is_reachable() {
    let groups = [
        [0u8, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ];
    let mut rgb = Vec::new();
    for (index, colour) in groups.iter().enumerate() {
        // Unequal multiplicities, so a wrong weight would show.
        for _ in 0..(200 + index * 37) {
            rgb.extend_from_slice(colour);
        }
    }
    let input = colours(&rgb);

    for k in 2..=groups.len() {
        let palette = colours(&kmeans_rgb(rgb.clone(), k, 100, Some(0.1)).unwrap());

        let mut mapped: Vec<[u8; 3]> = input.iter().map(|pixel| nearest(&palette, pixel)).collect();
        mapped.sort_unstable();
        mapped.dedup();

        assert_eq!(
            mapped.len(),
            palette.len(),
            "k={k}: palette {palette:?} has entries no pixel maps to"
        );
    }
}

/// With `k` equal to the number of distinct colours, every colour should get its
/// own centroid and the palette should reproduce the input exactly. This holds
/// because the seeds are distinct values, so each seed's own colour sits at
/// distance zero and keeps its cluster. A duplicated seed starves a cluster and
/// costs one of the `k` slots.
#[test]
fn a_palette_the_size_of_the_colour_count_reproduces_the_input() {
    let groups = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
    let mut rgb = Vec::new();
    for (index, colour) in groups.iter().enumerate() {
        for _ in 0..500 {
            // An unequal per-group offset, so a wrong weight would show.
            rgb.extend(colour.iter().map(|value| value.wrapping_add(index as u8)));
        }
    }
    let input = colours(&rgb);

    let palette = colours(&kmeans_rgb(rgb, groups.len(), 100, Some(0.1)).unwrap());
    assert_eq!(palette.len(), groups.len());

    for pixel in &input {
        assert_eq!(
            nearest(&palette, pixel),
            *pixel,
            "the palette should reproduce the input exactly"
        );
    }
}

/// With `k` at or below the number of distinct colours, every centroid is seeded
/// on an input pixel, that pixel is at distance zero from it, and no other
/// centroid is, so no cluster can empty. The palette must therefore be `k`
/// distinct colours. This is the property that makes an empty cluster, and the
/// NaN centroid that follows from one, unreachable.
#[test]
fn a_palette_at_or_below_the_colour_count_is_all_distinct_colours() {
    let groups = [
        [12u8, 40, 90],
        [200, 30, 60],
        [90, 210, 40],
        [240, 200, 10],
        [30, 30, 30],
    ];
    let mut rgb = Vec::new();
    for (index, colour) in groups.iter().enumerate() {
        for _ in 0..300 {
            rgb.extend(colour.iter().map(|value| value.wrapping_add(index as u8)));
        }
    }

    for k in 2..=groups.len() {
        let mut palette = colours(&kmeans_rgb(rgb.clone(), k, 100, Some(0.1)).unwrap());
        let total = palette.len();
        palette.sort_unstable();
        palette.dedup();

        assert_eq!(
            palette.len(),
            total,
            "k={k} with {} distinct colours: expected {total} distinct colours",
            groups.len()
        );
    }
}
