//! A differential check on `collapse`, which is the foundation everything else
//! stands on.
//!
//! `collapse` was refactored to build on a shared `Table::build` alongside
//! `PaletteMap`, and the golden fingerprint test does not cover it: that test
//! calls Hamerly directly, so a bug in the reduction would not show up there. The
//! existing tests check invariants, so this checks the *values* against a
//! deliberately naive implementation over many shapes.
//!
//! The comparison is over `u8` rows with `u32` counts rather than the `f64`
//! vectors the real function returns, because the weights are exact integer
//! counts and `f64` is not `Ord`.

use kmeans_wasm::packed_histogram::collapse;
use std::collections::HashMap;

/// A distinct value and how many times it occurs.
type Row = (Vec<u8>, u32);

fn as_rows(points: &[f64], weights: &[f64], components: usize) -> Vec<Row> {
    points
        .chunks_exact(components)
        .zip(weights)
        .map(|(row, weight)| {
            (
                row.iter().map(|value| *value as u8).collect(),
                *weight as u32,
            )
        })
        .collect()
}

/// The obvious implementation: count every pixel in a map, then list the
/// distinct values. Deliberately shares no code with the table.
fn reference(pixels: &[u8], components: usize) -> Option<(Vec<Row>, usize)> {
    if pixels.is_empty() {
        return None;
    }
    let mut counts: HashMap<Vec<u8>, u32> = HashMap::new();
    for pixel in pixels.chunks_exact(components) {
        *counts.entry(pixel.to_vec()).or_insert(0) += 1;
    }
    let distinct = counts.len();
    Some((counts.into_iter().collect(), distinct))
}

/// Compare as multisets, since the table hands rows back in slot order.
fn assert_same_rows(got: Vec<Row>, want: Vec<Row>, context: &str) {
    let mut got_sorted = got.clone();
    let mut want_sorted = want;
    got_sorted.sort();
    want_sorted.sort();
    assert_eq!(got_sorted, want_sorted, "{context}");
}

fn pixels_of(count: usize, components: usize, distinct: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(count * components);
    for index in 0..count {
        let pick = index % distinct;
        for channel in 0..components {
            pixels.push(((pick * 37 + channel * 11) % 256) as u8);
        }
    }
    pixels
}

#[test]
fn agrees_with_a_naive_count_on_values_and_weights() {
    let mut cases = 0;
    for components in [3usize, 4] {
        for count in [1usize, 2, 3, 17, 1000, 4096, 60_000] {
            for distinct in [1usize, 2, 3, 7, 64, 1000, 4096] {
                if distinct > count {
                    continue;
                }
                let context = format!("{count} px, {distinct} distinct, {components} components");
                let pixels = pixels_of(count, components, distinct);
                let got = collapse(&pixels, components);
                let want = reference(&pixels, components);
                assert_eq!(got.is_some(), want.is_some(), "{context}");
                let (Some(got), Some(want)) = (got, want) else {
                    continue;
                };
                assert_eq!(got.distinct, want.1, "{context}");
                assert_eq!(
                    got.weights.len(),
                    got.distinct,
                    "one weight per row: {context}"
                );
                assert_eq!(
                    got.points.len(),
                    got.distinct * components,
                    "row width: {context}"
                );
                assert_same_rows(
                    as_rows(&got.points, &got.weights, components),
                    want.0,
                    &context,
                );
                cases += 1;
            }
        }
    }
    assert!(
        cases > 50,
        "only {cases} cases ran, which is too few to be useful"
    );
}

#[test]
fn weights_always_sum_to_the_pixel_count() {
    for components in [3usize, 4] {
        for (count, distinct) in [(1usize, 1usize), (999, 3), (50_000, 100), (196_609, 1)] {
            let pixels = pixels_of(count, components, distinct);
            if let Some(found) = collapse(&pixels, components) {
                let total: f64 = found.weights.iter().sum();
                assert_eq!(
                    total, count as f64,
                    "{count} px, {distinct} distinct, {components} components"
                );
            }
        }
    }
}

/// Every distinct value represented exactly once, including the ones that look
/// like sentinels: zero, and the largest `u8`.
#[test]
fn covers_extreme_values_including_solid_black() {
    for components in [3usize, 4] {
        let mut pixels = vec![0u8; components * 10];
        for value in [255u8, 0, 128, 1] {
            let row = [value; 4];
            pixels.extend_from_slice(&row[..components]);
        }
        // Interleave, so the extremes are not all in one contiguous run.
        for value in [255u8, 0, 128, 1, 0, 255] {
            let row = [value; 4];
            pixels.extend_from_slice(&row[..components]);
        }

        let got = collapse(&pixels, components).expect("collapses");
        let want = reference(&pixels, components).expect("collapses");
        assert_eq!(got.distinct, want.1);
        assert_same_rows(
            as_rows(&got.points, &got.weights, components),
            want.0,
            "extremes",
        );
    }
}

/// A buffer that is not a whole number of pixels must not panic, and must not
/// claim more pixels than it was given.
#[test]
fn a_ragged_buffer_does_not_panic() {
    for components in [3usize, 4] {
        for length in 1..(components * 3) {
            let pixels = vec![7u8; length];
            if let Some(found) = collapse(&pixels, components) {
                let total: u32 = found.weights.iter().map(|w| *w as u32).sum();
                assert!(
                    total <= (length / components) as u32,
                    "length {length}, components {components}, total {total}"
                );
            }
        }
    }
}

/// An empty buffer, and a buffer of a single pixel, are the two ends of the range
/// and both are used in practice.
#[test]
fn the_ends_of_the_range() {
    for components in [3usize, 4] {
        assert!(collapse(&[], components).is_none(), "empty");
        let one = vec![9u8; components];
        let found = collapse(&one, components).expect("one pixel collapses");
        assert_eq!(found.distinct, 1);
        assert_eq!(found.weights, vec![1.0]);
        assert_eq!(found.points.len(), components);
    }
}

/// The exact boundary at which the table gives up. The load limit is three
/// quarters of the slot count, so 196,608 distinct values must still collapse and
/// 196,609 must not. An off-by-one either way would be silent.
#[test]
fn the_give_up_boundary_is_exactly_where_it_says() {
    // Enough pixels that the slot count saturates at 2^18, and exactly the
    // requested number of distinct values.
    fn with_distinct(distinct: usize) -> Option<usize> {
        let components = 3;
        let mut pixels = Vec::with_capacity(distinct * components);
        for index in 0..distinct {
            pixels.push((index % 256) as u8);
            pixels.push(((index / 256) % 256) as u8);
            pixels.push(((index / (256 * 256)) % 256) as u8);
        }
        collapse(&pixels, components).map(|found| found.distinct)
    }

    let limit = 196_608;
    assert_eq!(
        with_distinct(limit),
        Some(limit),
        "the limit itself must collapse"
    );
    assert_eq!(
        with_distinct(limit + 1),
        None,
        "one past the limit must not"
    );
    assert_eq!(
        with_distinct(limit - 1),
        Some(limit - 1),
        "one below the limit must"
    );
}
