//! When is mapping pixels onto a palette cheaper as a table than as a search?
//!
//! Mapping is `n * k` distance evaluations done per pixel, or `distinct * k` plus
//! `n` table probes if each distinct colour is resolved once. Which one wins is
//! decided by how much the buffer repeats, so input is generated with an exact
//! distinct count and the crossover is read off rather than guessed.
//!
//! Both variants run in this one binary and alternate, so drift hits them
//! equally. The table build is inside the timed region, because in production it
//! always is.
//!
//! Run with: cargo bench --bench mapping_probe

use kmeans_wasm::apply_palette;
use std::time::Instant;

const COMPONENTS: usize = 3;
const OUTPUT_COMPONENTS: usize = 4;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Exactly `distinct` colours, laid out in blocks 32 pixels wide so the buffer
/// has the spatial locality a photograph has. The block height is derived from
/// `distinct`, so the number of blocks that fit is `distinct` and the buffer
/// really does hold that many colours.
///
/// Both details of the layout matter more than they look. An earlier version of
/// this probe used `index % distinct`, which reaches every colour within the first
/// `distinct` pixels, so the running distinct rate hit its final value
/// immediately and the buffer had no locality at all. Every number taken from
/// that generator was about round-robin input rather than about images, which is
/// how a rule that gave up on the *running* distinct share came to be written and
/// then had to be taken out again.
///
/// A later version fixed the block at 32 by 32 without deriving the height, which
/// capped the real distinct count at `pixels / 1024` and quietly measured the same
/// 300-colour input on every row. The distinct count is therefore reported and
/// checked, not assumed.
fn pixels_with(count: usize, distinct: usize) -> (Vec<u8>, usize) {
    // Random triples collide by birthday, so at 4096 colours two of them are
    // already the same and the row silently measures one colour fewer. Rejecting
    // collisions costs a set of the colours and nothing else.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut palette: Vec<u8> = Vec::with_capacity(distinct * COMPONENTS);
    let mut taken = std::collections::HashSet::with_capacity(distinct * 2);
    while palette.len() < distinct * COMPONENTS {
        let mut triple = [0u8; COMPONENTS];
        for channel in &mut triple {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888974917);
            *channel = ((state >> 33) % 256) as u8;
        }
        if taken.insert(triple) {
            palette.extend_from_slice(&triple);
        }
    }

    // One block per colour, at most 32 wide, and short enough that `distinct`
    // blocks fit. Both dimensions have to be derived: fixing either one alone
    // caps the real distinct count and the row measures the wrong input.
    let per_colour = (count / distinct).max(1);
    let width = per_colour.min(32);
    let height = (per_colour / width).max(1);
    let block = width * height;
    let mut pixels = Vec::with_capacity(count * COMPONENTS);
    for index in 0..count {
        let pick = (index / block) % distinct;
        pixels.extend_from_slice(&palette[pick * COMPONENTS..][..COMPONENTS]);
    }

    let actual = {
        let mut seen = std::collections::HashSet::with_capacity(distinct);
        for pixel in pixels.as_chunks::<COMPONENTS>().0 {
            seen.insert([pixel[0], pixel[1], pixel[2]]);
        }
        seen.len()
    };
    (pixels, actual)
}

/// The variant a caller writes today: search the whole palette for every pixel.
fn search_per_pixel(pixels: &[u8], palette: &[u8]) -> Vec<u8> {
    let (pixels, _) = pixels.as_chunks::<COMPONENTS>();
    let (centroids, _) = palette.as_chunks::<COMPONENTS>();
    let mut output = vec![255u8; pixels.len() * OUTPUT_COMPONENTS];

    for (index, pixel) in pixels.iter().enumerate() {
        let mut nearest = 0usize;
        let mut nearest_distance = u32::MAX;
        for (entry, centroid) in centroids.iter().enumerate() {
            let mut distance = 0u32;
            for channel in 0..COMPONENTS {
                let delta = i32::from(pixel[channel]) - i32::from(centroid[channel]);
                distance += (delta * delta) as u32;
            }
            if distance < nearest_distance {
                nearest_distance = distance;
                nearest = entry;
            }
        }
        let base = nearest * COMPONENTS;
        let target = index * OUTPUT_COMPONENTS;
        output[target] = palette[base];
        output[target + 1] = palette[base + 1];
        output[target + 2] = palette[base + 2];
    }

    output
}

fn main() {
    let repeats = 7;
    let pixels = 307_200;
    let k = 32;

    println!(
        "{:>10} {:>7} {:>11} {:>10} {:>9} {:>6}",
        "distinct", "share", "search_ms", "table_ms", "speedup", "same"
    );

    for distinct in [
        64usize, 512, 4_096, 16_384, 32_768, 49_152, 65_536, 81_920, 98_304, 122_880, 140_714,
        196_608, 307_200,
    ] {
        let (input, actual) = pixels_with(pixels, distinct);
        assert_eq!(
            actual, distinct,
            "the generator must produce exactly the distinct count asked for, or the row means nothing"
        );
        // Real centroids sit close to the distinct values, so taking the palette
        // from the input is representative without depending on a clustering run.
        let palette: Vec<u8> = input.as_chunks::<COMPONENTS>().0[..k]
            .iter()
            .flatten()
            .copied()
            .collect();

        let mut search_times = Vec::new();
        for _ in 0..repeats {
            let started = Instant::now();
            let _ = search_per_pixel(&input, &palette);
            search_times.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        let search_ms = median(search_times);

        let mut table_times = Vec::new();
        let mut table_output = Vec::new();
        for _ in 0..repeats {
            let started = Instant::now();
            table_output = apply_palette(input.clone(), palette.clone(), COMPONENTS).unwrap();
            table_times.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        let table_ms = median(table_times);

        // The whole point of the table is that it is the same answer, so check it
        // on every row rather than once at the end.
        let same = table_output == search_per_pixel(&input, &palette);

        println!(
            "{distinct:>10} {:>6.1}% {search_ms:>11.2} {table_ms:>10.2} {:>8.2}x {:>6}",
            100.0 * distinct as f64 / pixels as f64,
            search_ms / table_ms,
            if same { "yes" } else { "NO" }
        );
    }
}
