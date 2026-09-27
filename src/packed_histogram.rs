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

/// A built open-addressing table over a packed buffer: which colours occur, and
/// how often.
///
/// `values` holds the occurrence count while the table is being built, and that
/// is what lets a zero mean "free slot" so solid black, key 0, is stored like
/// any other colour. `PaletteMap` then overwrites it with the palette entry plus
/// one. Both are non-zero for exactly the occupied slots, which is the single
/// property the probe depends on, so overwriting it costs no second array.
pub(crate) struct Table {
    /// The colour in each slot. Only meaningful where `values` is non-zero.
    keys: Vec<u32>,
    /// See the note on the struct.
    values: Vec<u32>,
    /// The probe sequence is `key * PHI >> shift`, masked to the slot count.
    shift: u32,
    mask: usize,
    distinct: usize,
}

impl Table {
    /// Builds a table over `slice`, or returns `None` when the buffer gives up.
    /// Callers read that as "this buffer does not repeat enough for a table to pay
    /// for itself" and take the direct route.
    ///
    /// The give-up is on the absolute distinct count, which bounds the work spent
    /// finding out. A criterion on the *share* of distinct values was tried and
    /// removed: the share seen so far is not a usable predictor of the final one,
    /// so any threshold misfired on real photographs. See AGENTS.md.
    ///
    /// `components` is 3 or 4.
    pub(crate) fn build(slice: &[u8], components: usize) -> Option<Self> {
        debug_assert!(matches!(components, 3 | 4));

        let pixel_count = slice.len() / components;
        if pixel_count == 0 {
            return None;
        }
        // Sized for the distinct count that could still be seen, so a buffer that
        // does collapse never probes twice.
        let slots = round_up_to_power_of_two(pixel_count.saturating_mul(2).min(MAX_SLOTS));
        let load_limit = slots / MAX_LOAD_DENOMINATOR * MAX_LOAD_NUMERATOR;
        let shift = 32 - slots.trailing_zeros();

        let mut keys = vec![0u32; slots];
        let mut values = vec![0u32; slots];
        let mut distinct = 0usize;

        for pixel in slice.chunks_exact(components) {
            let key = key_of(pixel);

            let mut index = (key.wrapping_mul(PHI) >> shift) as usize;
            loop {
                if values[index] == 0 {
                    keys[index] = key;
                    values[index] = 1;
                    distinct += 1;
                    if distinct > load_limit {
                        return None;
                    }
                    break;
                }
                if keys[index] == key {
                    values[index] += 1;
                    break;
                }
                index = (index + 1) & (slots - 1);
            }
        }

        Some(Table {
            keys,
            values,
            shift,
            mask: slots - 1,
            distinct,
        })
    }
}

/// Expands a key back into its components, in order.
fn unpack(key: u32, components: usize) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (component, byte) in out.iter_mut().enumerate().take(components) {
        *byte = (key >> ((components - 1 - component) * 8)) as u8;
    }
    out
}

/// The index of the palette entry nearest to a pixel's colour. `palette` is
/// packed with the same `components` as the pixel.
///
/// Squared Euclidean, summed in ascending order. The comparison is strict, so a
/// tie keeps the lower index, which is what the clustering core does and what a
/// caller writing this loop in JavaScript ends up with.
///
/// The arithmetic is integer, and that is not a shortcut. A channel difference is
/// at most 255, its square at most 65025, and four of those sum to 260100, so
/// `u32` is exact and reproduces the `f64` a JavaScript caller would compute, bit
/// for bit.
pub(crate) fn nearest_palette(pixel: &[u8], palette: &[u8], components: usize) -> usize {
    let entries = palette.len() / components;
    let mut nearest = 0;
    let mut nearest_distance = u32::MAX;

    for entry in 0..entries {
        let base = entry * components;
        let mut distance = 0u32;
        for component in 0..components {
            let difference = i32::from(pixel[component]) - i32::from(palette[base + component]);
            distance += (difference * difference) as u32;
        }
        if distance < nearest_distance {
            nearest_distance = distance;
            nearest = entry;
        }
    }

    nearest
}

/// Collapses `slice` into distinct values with weights, or returns `None` when the
/// input repeats too little for the reduction to pay.
///
/// `components` is 3 or 4.
pub fn collapse(slice: &[u8], components: usize) -> Option<Collapsed> {
    let table = Table::build(slice, components)?;
    let distinct = table.distinct;

    // The table is walked in slot order, so rows come out grouped by hash rather
    // than in first-seen order. That is fine: the reduction is exact whatever
    // order the distinct values are listed in, and the seed draw is weighted, so
    // it does not depend on the order either.
    let mut points = Vec::with_capacity(distinct * components);
    let mut weights = Vec::with_capacity(distinct);
    for (key, count) in table.keys.iter().zip(table.values.iter()) {
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

/// A colour to the palette entry it becomes, resolved once per distinct colour.
///
/// Mapping is the other half of colour quantization and callers write it
/// themselves. Searching the palette for every pixel is `n * k`, which for a
/// megapixel at 32 colours is 29 million distance evaluations. Resolving each
/// distinct colour once is `distinct * k`, and real photographs repeat heavily:
/// the standard test images here are 6% to 46% distinct.
///
/// Measured with `benches/mapping_probe.rs` against a per-pixel search at k=32,
/// the table is 5.9x on input that is 0.02% distinct, 2.5x at 11%, 1.5x at 32%,
/// level at 46%, and 0.8x at 64%. So it wins while the buffer is under about half
/// distinct, which is where photographs and flat-region images sit. The other
/// side is bounded rather than open-ended, because `Table::build` gives up once
/// the distinct count exceeds the table: input that does not repeat at all costs
/// 0.83x, which is the wasted build plus a plain search.
pub(crate) struct PaletteMap {
    table: Table,
}

impl PaletteMap {
    /// Builds a map over `slice`, or returns `None` when the buffer holds more
    /// distinct values than the table can. The caller should then search the
    /// palette per pixel, which is slower but always correct.
    ///
    /// `palette` is the centroids from a clustering entry point, packed with the
    /// same `components` as the buffer.
    pub(crate) fn build(slice: &[u8], components: usize, palette: &[u8]) -> Option<Self> {
        if palette.is_empty() {
            return None;
        }
        let mut table = Table::build(slice, components)?;

        // Resolve every distinct colour to its nearest entry, once. Occupied
        // slots hold a count of at least 1, so entry plus one is never zero and
        // the occupancy test still holds.
        for index in 0..table.keys.len() {
            if table.values[index] != 0 {
                let entry =
                    nearest_palette(&unpack(table.keys[index], components), palette, components);
                table.values[index] = entry as u32 + 1;
            }
        }

        Some(PaletteMap { table })
    }

    /// The palette entry for one pixel. Must be called with the same `components`
    /// the map was built for, on a pixel from the buffer it was built over.
    pub(crate) fn lookup(&self, pixel: &[u8]) -> usize {
        let key = key_of(pixel);
        let table = &self.table;

        let mut index = (key.wrapping_mul(PHI) >> table.shift) as usize;
        loop {
            // The key has to be compared, not just the occupancy, or a collision
            // would hand back whichever colour was probed first.
            if table.keys[index] == key {
                return (table.values[index] - 1) as usize;
            }
            if table.values[index] == 0 {
                // Absent, which cannot happen for a pixel the table was built
                // over. Returning the first entry beats wrapping in a full table.
                return 0;
            }
            index = (index + 1) & table.mask;
        }
    }
}
