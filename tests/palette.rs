//! Mapping pixels onto a palette.
//!
//! The property that matters is not speed but agreement: `apply_palette` must
//! return byte-for-byte what a caller gets from searching the palette per pixel,
//! because that is what callers do today and any difference would show up as a
//! subtly wrong image rather than an error.

use kmeans_wasm::apply_palette;

/// A deliberately naive reference: search the whole palette for every pixel,
/// with a strict `<` so a tie keeps the lower index. This is the loop a caller
/// writes in JavaScript, transcribed.
fn reference(pixels: &[u8], palette: &[u8], components: usize) -> Vec<u8> {
    let pixel_count = pixels.len() / components;
    let mut output = vec![255u8; pixel_count * 4];

    for (index, pixel) in pixels.chunks_exact(components).enumerate() {
        let mut nearest = 0usize;
        let mut nearest_distance = u64::MAX;
        for (entry, centroid) in palette.chunks_exact(components).enumerate() {
            let mut distance = 0u64;
            for channel in 0..components {
                let delta = i64::from(pixel[channel]) - i64::from(centroid[channel]);
                distance += (delta * delta) as u64;
            }
            if distance < nearest_distance {
                nearest_distance = distance;
                nearest = entry;
            }
        }
        let base = nearest * components;
        let target = index * 4;
        output[target] = palette[base];
        output[target + 1] = palette[base + 1];
        output[target + 2] = palette[base + 2];
        output[target + 3] = if components == 4 {
            palette[base + 3]
        } else {
            255
        };
    }

    output
}

/// Deterministic distinct colours, including ties by construction.
fn palette_of(count: usize, components: usize) -> Vec<u8> {
    let mut palette = Vec::with_capacity(count * components);
    for entry in 0..count {
        for channel in 0..components {
            palette.push(((entry * 37 + channel * 11) % 256) as u8);
        }
    }
    palette
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
fn matches_a_per_pixel_search_over_many_shapes() {
    for components in [3usize, 4] {
        for (pixels_count, distinct) in [
            (1usize, 1usize),
            (2, 1),
            (2, 2),
            (7, 3),
            (100, 5),
            (4_096, 40),
            (4_096, 4_096),
            (4_096, 3),
        ] {
            for entries in [1usize, 2, 3, 8, 64] {
                let palette = palette_of(entries, components);
                let pixels = pixels_of(pixels_count, components, distinct);
                let got = apply_palette(pixels.clone(), palette.clone(), components).unwrap();
                let want = reference(&pixels, &palette, components);
                assert_eq!(
                    got, want,
                    "{pixels_count} px, {distinct} distinct, {entries} entries, {components} components"
                );
            }
        }
    }
}

#[test]
fn ties_resolve_to_the_lower_index() {
    // Two identical entries, so every pixel is exactly equidistant and the strict
    // comparison must pick the first.
    let palette = vec![10u8, 20, 30, 10, 20, 30];
    let pixels = vec![10u8, 20, 30, 200, 100, 50, 11, 20, 30];

    let mapped = apply_palette(pixels, palette, 3).unwrap();

    // Entries 0 and 1 are identical, so the first three channels are entry 0's.
    assert_eq!(&mapped[0..3], &[10, 20, 30]);
    // The middle pixel is nearer entry 1, at distance 0 versus a squared
    // distance of 3*190*190, so it maps to entry 1, which is the same colours.
    assert_eq!(&mapped[4..7], &[10, 20, 30]);
    // The last pixel is one unit away in red, so still entry 0.
    assert_eq!(&mapped[8..11], &[10, 20, 30]);
}

#[test]
fn alpha_is_carried_through_and_synthesised() {
    // Distinct colours per alpha, so nothing is a tie and each pixel has exactly
    // one nearest entry. A tie would send both pixels to the lower index, which
    // is correct but would not test alpha at all.
    let palette = vec![10u8, 20, 30, 0, 90, 80, 70, 255];
    let pixels = vec![10u8, 20, 30, 0, 90, 80, 70, 255];

    let mapped = apply_palette(pixels, palette, 4).unwrap();
    assert_eq!(&mapped[0..4], &[10, 20, 30, 0], "transparent stays");
    assert_eq!(&mapped[4..8], &[90, 80, 70, 255], "opaque stays");

    // From an RGB source the alpha is always opaque, never read from the palette.
    let rgb_palette = vec![10u8, 20, 30];
    let mapped = apply_palette(vec![10u8, 20, 30], rgb_palette, 3).unwrap();
    assert_eq!(&mapped, &[10, 20, 30, 255]);
}

#[test]
fn output_is_always_rgba() {
    let palette = vec![0u8, 0, 0, 0, 0, 0];
    let rgb = apply_palette(vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9], palette.clone(), 3).unwrap();
    assert_eq!(rgb.len(), 3 * 4, "three pixels, four values each");

    let rgba_palette = vec![0u8, 0, 0, 0, 0, 0, 0, 0];
    let rgba = apply_palette(vec![1u8, 2, 3, 4], rgba_palette, 4).unwrap();
    assert_eq!(rgba.len(), 4);
}

#[test]
fn solid_black_is_a_key_not_an_empty_slot() {
    // Key zero is what a naive table would use for an empty slot, and black is a
    // real colour, so it has to be counted like any other.
    let palette = vec![255u8, 255, 255, 0, 0, 0];
    let mut pixels = vec![0u8; 3 * 900];
    pixels.extend_from_slice(&[255, 255, 255, 255, 255, 255, 255, 255, 255]);

    let mapped = apply_palette(pixels.clone(), palette.clone(), 3).unwrap();
    assert_eq!(mapped, reference(&pixels, &palette, 3));
    // The black pixels map to the black entry, the white ones to the white entry.
    assert_eq!(&mapped[0..3], &[0, 0, 0]);
    assert_eq!(&mapped[mapped.len() - 3..], &[255, 255, 255]);
}

// Rejected input builds a `JsValue`, which aborts on native targets, so those
// cases live in `tests/web.rs`.

#[test]
fn an_empty_buffer_maps_to_an_empty_buffer() {
    let palette = vec![0u8, 0, 0];
    assert_eq!(
        apply_palette(Vec::new(), palette, 3).unwrap(),
        Vec::<u8>::new()
    );
}
