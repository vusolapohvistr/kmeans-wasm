//! How much of a Hamerly round is the distance pass, and how much is the bound
//! bookkeeping around it?
//!
//! This is the question a GPU split turns on. Distances are order-independent
//! and embarrassingly parallel, so they can move to a shader and still be
//! bit-identical. The bound tests, the bound updates and the cluster sums cannot:
//! the first two are loop-carried across points, and the sums are summed in
//! ascending point order, which is the whole reason the centroids are stable.
//!
//! So the ceiling on any GPU split is set by the ratio between the distances and
//! everything else. Counts rather than timings, because counts are exact.
//!
//! Run with: cargo bench --bench work_split --features counters

use kmeans_wasm::kmeans_triangle::{Optional, hamerly_kmeans_dispatched_weighted};
use kmeans_wasm::packed_histogram::collapse;
use kmeans_wasm::work;

const COMPONENTS: usize = 3;

/// Exactly `distinct` colours in contiguous blocks, so the distinct share is the
/// only thing that varies. The block layout matters: round-robin input has no
/// locality and every measurement taken from it describes the wrong thing.
fn pixels_with(count: usize, distinct: usize) -> Vec<u8> {
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

    let per_colour = (count / distinct).max(1);
    let width = per_colour.min(32);
    let height = (per_colour / width).max(1);
    let block = width * height;

    let mut pixels = Vec::with_capacity(count * COMPONENTS);
    for index in 0..count {
        let pick = (index / block) % distinct;
        pixels.extend_from_slice(&palette[pick * COMPONENTS..][..COMPONENTS]);
    }
    pixels
}

fn main() {
    let k = 32;
    let pixels = 307_200;

    println!(
        "{:>9} {:>7} {:>8} {:>7} {:>11} {:>10} {:>10} {:>9} {:>9} {:>8}",
        "distinct",
        "share",
        "rounds",
        "points",
        "dist_evals",
        "per_round",
        "bound_ops",
        "sum_ops",
        "full_scan",
        "gpu_share"
    );

    for distinct in [4_096usize, 16_384, 32_768, 65_536, 98_304, 140_714, 196_608] {
        let input = pixels_with(pixels, distinct);
        // Above 196,608 distinct the histogram gives up and the run is uncollapsed,
        // which is a different question and is what MAX_SLOTS is for.
        let Some(collapsed) = collapse(&input, COMPONENTS) else {
            println!("{distinct:>9} exceeds the table; the reduction would not apply");
            continue;
        };

        work::reset();
        let result = hamerly_kmeans_dispatched_weighted(
            k,
            100,
            0.1,
            &collapsed.points,
            COMPONENTS,
            Optional {
                weights: Some(&collapsed.weights),
                ..Optional::default()
            },
        );
        let counts = work::snapshot();
        let rounds = result.iterations.max(1) as u64;
        let points = collapsed.distinct as u64;

        // A distance evaluation is three subtracts, three multiplies and two adds
        // on f64, plus the loop and the load. A bound update is two loads, an add
        // and a store. A sum update is a multiply-add into a running total, and
        // carries a dependency chain. These weights are a model, and the point of
        // the column is that it is an argument and not a measurement.
        const COST_DISTANCE: f64 = 8.0;
        const COST_BOUND: f64 = 3.0;
        const COST_SUM: f64 = 2.0;

        let distance_work = counts.distance_evaluations as f64 * COST_DISTANCE;
        let bound_work = (counts.bound_checks + counts.bound_updates) as f64 * COST_BOUND;
        let sum_work = counts.sum_updates as f64 * COST_SUM;
        let gpu_share = distance_work / (distance_work + bound_work + sum_work);

        println!(
            "{distinct:>9} {:>6.1}% {rounds:>8} {points:>7} {:>11} {:>10.0} {:>10} {:>10} {:>9} {:>8.0}% {:>8.0}%",
            100.0 * distinct as f64 / pixels as f64,
            counts.distance_evaluations,
            counts.distance_evaluations as f64 / rounds as f64,
            counts.bound_checks + counts.bound_updates,
            counts.sum_updates,
            counts.full_scans,
            100.0 * counts.full_scans as f64 / rounds as f64 / points as f64,
            100.0 * gpu_share,
        );
    }

    println!(
        "\nper_round is distances per point per round, and it is the number that\n\
         decides everything: at 1.0 a round costs n distances, at 32.0 it costs n * k.\n\
         gpu_share is a model with stated weights, not a measurement."
    );
}
