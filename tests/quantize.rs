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

/// Corner cases for the reduction to distinct colours, which now sits on the
/// packed path and runs before anything else does.
mod distinct_colours {
    use super::{colours, kmeans_rgb, kmeans_rgba, nearest, squared_distance};

    /// A degenerate but well-formed one pixel. The reduction collapses it to a
    /// single weighted value, so this is the smallest input that reaches the
    /// histogram at all.
    ///
    /// Malformed lengths are rejected by building a `JsValue`, which aborts on
    /// native targets, so those cases live in `tests/web.rs` instead.
    #[test]
    fn a_single_pixel() {
        let one = kmeans_rgb(vec![10, 20, 30], 2, 100, Some(0.1)).unwrap();
        assert_eq!(one.len(), 6);
        assert!(one.iter().all(|value| matches!(*value, 10 | 20 | 30)));
    }

    /// Two pixels and the maximum `k` the API allows. Every cluster must still
    /// hold something, or the caller gets a palette entry nothing maps to.
    #[test]
    fn more_clusters_than_pixels() {
        let rgb = vec![10, 20, 30, 200, 100, 50];
        let palette = colours(&kmeans_rgb(rgb.clone(), 8, 100, Some(0.1)).unwrap());
        assert_eq!(palette.len(), 8);

        let input = colours(&rgb);
        let mut mapped: Vec<[u8; 3]> = input.iter().map(|pixel| nearest(&palette, pixel)).collect();
        mapped.sort_unstable();
        mapped.dedup();
        assert_eq!(mapped.len(), 2, "both pixels must be represented");
    }

    /// Every pixel identical, with `k` above the single distinct value.
    #[test]
    fn one_colour_with_more_clusters_requested() {
        for fill in [0u8, 7, 128, 255] {
            let rgb = vec![fill; 3 * 2048];
            let palette = kmeans_rgb(rgb, 16, 100, Some(0.1)).unwrap();
            assert_eq!(palette.len(), 48);
            assert!(
                palette.iter().all(|value| *value == fill),
                "a solid {fill} image must quantize to solid {fill}, got {palette:?}"
            );
        }
    }

    /// Alpha is clustered like any other component, so a fully transparent and a
    /// fully opaque region of the same colour must stay apart.
    #[test]
    fn alpha_survives_the_reduction() {
        let mut rgba = Vec::new();
        for _ in 0..500 {
            rgba.extend_from_slice(&[10, 20, 30, 0]);
        }
        for _ in 0..500 {
            rgba.extend_from_slice(&[10, 20, 30, 255]);
        }

        let palette = kmeans_rgba(rgba, 2, 100, Some(0.1)).unwrap();
        let entries: Vec<[u8; 4]> = palette.as_chunks::<4>().0.to_vec();
        assert_eq!(entries.len(), 2);

        let mut alphas: Vec<u8> = entries.iter().map(|entry| entry[3]).collect();
        alphas.sort_unstable();
        assert_eq!(alphas, vec![0, 255], "both alpha values must be present");
        for entry in &entries {
            assert_eq!(&entry[..3], &[10, 20, 30], "the colour must survive too");
        }
    }

    /// Solid black is key zero, which is also the marker a naive table would use
    /// for an empty slot. It must be counted like any other colour.
    #[test]
    fn solid_black_is_a_real_colour() {
        let mut rgb = Vec::new();
        for _ in 0..900 {
            rgb.extend_from_slice(&[0, 0, 0]);
        }
        for _ in 0..100 {
            rgb.extend_from_slice(&[255, 255, 255]);
        }

        let palette = colours(&kmeans_rgb(rgb.clone(), 2, 100, Some(0.1)).unwrap());
        assert_eq!(palette.len(), 2);
        assert!(palette.contains(&[0, 0, 0]), "black must be present");
        assert!(palette.contains(&[255, 255, 255]), "white must be present");

        // And it must be the nearest entry to its own pixels.
        for pixel in colours(&rgb) {
            let cost = nearest(&palette, &pixel);
            assert_eq!(squared_distance(&pixel, &cost), 0.0);
        }
    }

    /// Two colours that differ in only one channel, the case where a palette
    /// entry can be nearest to nothing.
    #[test]
    fn colours_one_unit_apart() {
        let mut rgb = Vec::new();
        for index in 0..600 {
            rgb.extend_from_slice(&[100, 100, 100 + (index % 2) as u8]);
        }

        let input = colours(&rgb);
        let palette = colours(&kmeans_rgb(rgb, 2, 100, Some(0.1)).unwrap());
        let mut mapped: Vec<[u8; 3]> = input.iter().map(|pixel| nearest(&palette, pixel)).collect();
        mapped.sort_unstable();
        mapped.dedup();
        assert_eq!(
            mapped.len(),
            palette.len(),
            "with k=2 over two colours every entry must be reachable"
        );
    }
}
