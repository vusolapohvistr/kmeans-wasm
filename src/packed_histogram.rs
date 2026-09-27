//! Collapse a packed colour buffer to its distinct values with weights.
//!
//! Equal pixels are interchangeable: they take the same cluster and add the same
//! amount to that cluster's sum, so clustering the distinct values once, with a
//! weight equal to the number of occurrences, gives the same clustering as
//! clustering every pixel. Celebi's Weighted Sort-Means uses exactly this
//! reduction and reports identical results to plain k-means given the same
//! initial centres.
//!
//! The saving is the factor by which the point set shrinks. For the standard
//! images in that literature it is 1.7x to 14x, and flat-region input, which is
//! what canvas, PNG, screenshots and icons produce, can be far more.
//!
//! The cost is building the histogram. On input that barely repeats there is
//! nothing to collapse, so the table is given up part way through rather than
//! filled. An earlier version tried to avoid that cost with a sample of the
//! buffer first, and it was wrong: an evenly spaced sample is aliased to zero by
//! input that repeats with a long period, and reported no duplicates at all for
//! input that was 75% duplicate. No sampling scheme catches that, so the bounded
//! table does the work instead and the threshold is on distinct count alone.

/// Slots are a power of two, so the probe sequence is `key * PHI >> shift`.
const PHI: u32 = 2_654_435_761;

/// The table never grows, so the work spent discovering there is nothing to
/// collapse is bounded by the table rather than by the input. That bound is the
/// whole cost of trying, and it is what makes the reduction safe to attempt
/// unconditionally: on input with no duplicates the run gives up after
/// `MAX_SLOTS * MAX_LOAD_NUMERATOR / MAX_LOAD_DENOMINATOR` insertions, which is
/// 196,608 of them.
///
/// The size is set by the real photographs, not by taste. At 2^17 the table held
/// 98,304 distinct values and the reduction declined two of four test images that
/// would clearly have benefited: a 307,200 pixel photograph at 45.8% distinct
/// took 578 ms at k=32. Raising it to 2^18 brought that to 341 ms and a 36.6%
/// distinct one from 581 ms to 269 ms, while the images that already collapsed
/// were unchanged.
///
/// The bail-out is still free at that size. Measured over 480,000 pixels from
/// 100% distinct down to 50%, the reduced path came out at 0.99x to 1.00x, inside
/// this machine's noise, and the reduction took over from 25% distinct at 4.9x
/// and reached 12.7x at 10% distinct and 50x at 1%.
const MAX_SLOTS: usize = 1 << 18;

/// Occupancy above which the reduction is abandoned part way through. Probing
/// degrades sharply past about three quarters full.
const MAX_LOAD_NUMERATOR: usize = 3;
const MAX_LOAD_DENOMINATOR: usize = 4;

/// The distinct values, widened to `f64`, with a weight per value.
pub struct Collapsed {
    /// One row per distinct colour, `components` values each, row-major.
    pub points: Vec<f64>,
    /// How many pixels each row stands for. Always at least 1.
    pub weights: Vec<f64>,
    /// Distinct values found. Equal to `weights.len()`.
    pub distinct: usize,
}

/// Packs a pixel's components into a key. Exact for four bytes.
fn key_of(pixel: &[u8]) -> u32 {
    let mut key = 0u32;
    for component in pixel {
        key = (key << 8) | u32::from(*component);
    }
    key
}

fn round_up_to_power_of_two(value: usize) -> usize {
    let mut slots = 1;
    while slots < value {
        slots <<= 1;
    }
    slots
}

/// Collapses `slice` into distinct values with weights, or returns `None` when
/// the input repeats too little for the reduction to pay.
///
/// `components` is 3 or 4.
pub fn collapse(slice: &[u8], components: usize) -> Option<Collapsed> {
    debug_assert!(matches!(components, 3 | 4));

    let pixel_count = slice.len() / components;
    if pixel_count == 0 {
        return None;
    }
    // Sized for the distinct count that could still be seen, so an input that
    // does collapse never probes twice.
    let slots = round_up_to_power_of_two(pixel_count.saturating_mul(2).min(MAX_SLOTS));
    let load_limit = slots / MAX_LOAD_DENOMINATOR * MAX_LOAD_NUMERATOR;
    let shift = 32 - slots.trailing_zeros();

    // A count of zero marks a free slot, so the key needs no sentinel and solid
    // black, key 0, is stored like any other colour.
    let mut keys = vec![0u32; slots];
    let mut counts = vec![0u32; slots];
    let mut distinct = 0usize;

    for pixel in slice.chunks_exact(components) {
        let key = key_of(pixel);

        let mut index = (key.wrapping_mul(PHI) >> shift) as usize;
        loop {
            if counts[index] == 0 {
                keys[index] = key;
                counts[index] = 1;
                distinct += 1;
                if distinct > load_limit {
                    return None;
                }
                break;
            }
            if keys[index] == key {
                counts[index] += 1;
                break;
            }
            index = (index + 1) & (slots - 1);
        }
    }

    // The table is walked in slot order, so rows come out grouped by hash rather
    // than in first-seen order. That is fine: the reduction is exact whatever
    // order the distinct values are listed in, and the seed draw is weighted, so
    // it does not depend on the order either.
    let mut points = Vec::with_capacity(distinct * components);
    let mut weights = Vec::with_capacity(distinct);
    for (key, count) in keys.iter().zip(counts.iter()) {
        if *count == 0 {
            continue;
        }
        // The key is the pixel's bytes in order, so the first component is the
        // most significant byte.
        for component in (0..components).rev() {
            points.push(f64::from((key >> (component * 8)) as u8));
        }
        weights.push(f64::from(*count));
    }

    Some(Collapsed {
        points,
        weights,
        distinct,
    })
}
