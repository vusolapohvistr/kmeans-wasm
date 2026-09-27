//! Where does the reduction to distinct colours start paying for itself?
//!
//! Input is generated with an exact distinct-colour count, so the crossover can
//! be read off rather than guessed. The histogram build is inside the timed
//! region, because in production it always is.

use kmeans_wasm::kmeans_triangle::{Optional, hamerly_kmeans_dispatched_weighted};
use kmeans_wasm::packed_histogram::collapse;
use std::time::Instant;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn as_f64(slice: &[u8]) -> Vec<f64> {
    slice.iter().map(|value| f64::from(*value)).collect()
}

/// Exactly `distinct` colours, each used in proportion to its index so regions
/// differ in size, laid out round-robin.
fn palette_pixels(pixels: usize, distinct: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut palette = Vec::with_capacity(distinct * 3);
    for _ in 0..distinct * 3 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        palette.push(((state >> 33) % 256) as u8);
    }
    let mut data = Vec::with_capacity(pixels * 3);
    for index in 0..pixels {
        let pick = (index % distinct) * 3;
        data.extend_from_slice(&palette[pick..pick + 3]);
    }
    data
}

fn main() {
    let pixels = 480_000;
    let repeats = 7;
    let k = 32;

    println!("Crossover, {pixels} pixels, k={k}, median of {repeats}\n");
    println!(
        "{:>10}  {:>8}  {:>10}  {:>10}  {:>8}",
        "distinct", "share", "full (ms)", "reduced (ms)", "speedup"
    );

    for percent in [100usize, 95, 90, 80, 70, 50, 25, 10, 1, 0] {
        let distinct = if percent == 0 {
            1
        } else {
            pixels * percent / 100
        };
        let data = palette_pixels(pixels, distinct);
        let wide = as_f64(&data);
        let collapses = collapse(&data, 3).is_some();

        let mut full_ms = Vec::new();
        for _ in 0..repeats {
            let started = Instant::now();
            let _ = hamerly_kmeans_dispatched_weighted(k, 100, 0.0, &wide, 3, Optional::default());
            full_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }

        let mut reduced_ms = Vec::new();
        for _ in 0..repeats {
            let started = Instant::now();
            match collapse(&data, 3) {
                Some(ref c) => {
                    let _ = hamerly_kmeans_dispatched_weighted(
                        k,
                        100,
                        0.0,
                        &c.points,
                        3,
                        Optional {
                            weights: Some(&c.weights),
                            ..Optional::default()
                        },
                    );
                }
                None => {
                    let _ = hamerly_kmeans_dispatched_weighted(
                        k,
                        100,
                        0.0,
                        &wide,
                        3,
                        Optional::default(),
                    );
                }
            }
            reduced_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }

        let full = median(full_ms);
        let reduced = median(reduced_ms);
        println!(
            "{distinct:>10}  {:>7.1}%  {full:>10.2}  {reduced:>10.2}  {:>7.2}x  {}",
            100.0 * distinct as f64 / pixels as f64,
            full / reduced,
            if collapses { "collapsed" } else { "declined" }
        );
    }
}
