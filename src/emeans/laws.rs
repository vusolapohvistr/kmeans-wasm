//! Scaling laws and schedule helpers for e-means.

/// The cosine schedule ends at one tenth of the initial step.
const ALPHA_MIN_FACTOR: f64 = 0.1;

/// The packed-paths batch law uses the smaller "quality" batch up to this many
/// dimensions; above that it uses the throughput law.
const QUALITY_BATCH_MAX_DIMENSIONS: usize = 16;

// Batch law B = C * k^0.134749 * n^0.876473 * T^0.813344. The quality
// coefficient is one eighth of the throughput one, matching the reference
// implementation's shipped fits (which differ slightly from the paper's
// rounded table values).
const BATCH_QUALITY_COEFFICIENT: f64 = 0.004_391_75;
const BATCH_THROUGHPUT_COEFFICIENT: f64 = 0.035_134;
const BATCH_K_EXPONENT: f64 = 0.134_749;
const BATCH_N_EXPONENT: f64 = 0.876_473;
const BATCH_T_EXPONENT: f64 = 0.813_344;

// Step law alpha0 = C * k^-0.4617 * B^0.9003 * n^-0.4498 * T^-0.1057.
const ALPHA_COEFFICIENT: f64 = 0.824_4;
const ALPHA_K_EXPONENT: f64 = -0.461_7;
const ALPHA_B_EXPONENT: f64 = 0.900_3;
const ALPHA_N_EXPONENT: f64 = -0.449_8;
const ALPHA_T_EXPONENT: f64 = -0.105_7;
const ALPHA_MINIMUM: f64 = 1e-4;

/// The batch size law. `n` is the number of points, not rows, so a collapsed
/// histogram uses the same statistical batch as the pixels it stands for.
pub(super) fn resolve_batch_size(
    n: f64,
    k: usize,
    epochs: usize,
    dimensions: usize,
    given: Option<usize>,
) -> usize {
    let n = n.max(1.0);
    let point_total = n.ceil().min(usize::MAX as f64) as usize;

    if let Some(batch_size) = given {
        // The batch is measured in points, not rows, so a collapsed histogram
        // may still fill a batch larger than its row count from weighted rows.
        return batch_size.clamp(1, point_total);
    }

    let coefficient = if dimensions <= QUALITY_BATCH_MAX_DIMENSIONS {
        BATCH_QUALITY_COEFFICIENT
    } else {
        BATCH_THROUGHPUT_COEFFICIENT
    };
    let raw = coefficient
        * (k.max(1) as f64).powf(BATCH_K_EXPONENT)
        * n.powf(BATCH_N_EXPONENT)
        * (epochs.max(1) as f64).powf(BATCH_T_EXPONENT);

    let rounded = round_to_power_of_two(raw);
    let cap = if dimensions <= QUALITY_BATCH_MAX_DIMENSIONS {
        (n / 2.0) as usize
    } else {
        round_to_power_of_two(n / 2.0)
    };
    rounded.clamp(1, point_total).min(cap.max(1))
}

/// The initial EMA step law, clamped to the range the reference implementation
/// uses so a tiny or enormous problem cannot produce a step that stalls or
/// overshoots.
pub(super) fn resolve_alpha0(n: f64, k: usize, epochs: usize, batch_size: usize) -> f64 {
    let alpha = ALPHA_COEFFICIENT
        * (k.max(1) as f64).powf(ALPHA_K_EXPONENT)
        * (batch_size.max(1) as f64).powf(ALPHA_B_EXPONENT)
        * n.max(1.0).powf(ALPHA_N_EXPONENT)
        * (epochs.max(1) as f64).powf(ALPHA_T_EXPONENT);
    alpha.clamp(ALPHA_MINIMUM, 1.0)
}

/// Cosine-anneal the step from `alpha0` at the first update to a tenth of it at
/// the last, as in Equation 5 of the paper.
pub(super) fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = (step.min(total_steps) as f64) / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Round a positive finite value to the nearest power of two, with the geometric
/// mean as the tie break. Values at or below one round to one.
pub(super) const fn round_to_power_of_two(value: f64) -> usize {
    if !value.is_finite() || value <= 1.0 {
        return 1;
    }
    // Clearing the mantissa leaves the largest power of two not above `value`.
    // The cast saturates for values beyond `usize`, which cannot occur for a
    // point count but keeps this total.
    let low = f64::from_bits(value.to_bits() & 0x7ff0_0000_0000_0000);
    let high = low * 2.0;
    let rounded = if value / low < high / value {
        low
    } else {
        high
    };
    rounded as usize
}
