//! Update phase: the EMA centroid updates and revival tools used by e-means.

use crate::emeans::util::Scramble;

/// Packs row-major `f64` centroids into 2-lane panels (`k_pairs * dimensions * 2`),
/// padding any odd trailing centroid slot with `f64::INFINITY`.
pub(super) fn pack_centroids_f64_panels(
    centroids: &[f64],
    dimensions: usize,
    k: usize,
    panels: &mut [f64],
) {
    let k_pairs = k.div_ceil(2);
    panels.fill(f64::INFINITY);
    for pair in 0..k_pairs {
        let c0 = pair * 2;
        let c1 = c0 + 1;
        let base = pair * dimensions * 2;
        let row0 = &centroids[c0 * dimensions..(c0 + 1) * dimensions];
        if c1 < k {
            let row1 = &centroids[c1 * dimensions..(c1 + 1) * dimensions];
            for part in 0..dimensions {
                panels[base + part * 2] = row0[part];
                panels[base + part * 2 + 1] = row1[part];
            }
        } else {
            for part in 0..dimensions {
                panels[base + part * 2] = row0[part];
            }
        }
    }
}

/// Packs row-major `f64` centroids into 4-lane `f32` panels (`k_panels * dimensions * 4`),
/// padding unused cluster lanes with `f32::INFINITY`.
pub(super) fn pack_centroids_f32_panels(centroids: &[f64], dimensions: usize, panels: &mut [f32]) {
    panels.fill(f32::INFINITY);
    for (cluster, row) in centroids.chunks_exact(dimensions).enumerate() {
        let panel = (cluster >> 2) * dimensions * 4;
        let lane = cluster & 3;
        for (part, value) in row.iter().enumerate() {
            panels[panel + part * 4 + lane] = *value as f32;
        }
    }
}

/// Packs row-major `f64` centroids for 3D or 4D `f32x4` dot-product searches into
/// contiguous 16-float (`[cx0..3, cy0..3, cz0..3, hns0..3]`) or 20-float panels.
pub(super) fn pack_centroids_f32_dot_panels(
    centroids: &[f64],
    dimensions: usize,
    panels: &mut [f32],
) {
    let stride = (dimensions + 1) * 4;
    panels.fill(f32::INFINITY);
    for (cluster, row) in centroids.chunks_exact(dimensions).enumerate() {
        let panel = (cluster >> 2) * stride;
        let lane = cluster & 3;
        let mut norm_sq = 0.0f32;
        for (part, value) in row.iter().enumerate() {
            let v = *value as f32;
            panels[panel + part * 4 + lane] = v;
            norm_sq += v * v;
        }
        panels[panel + dimensions * 4 + lane] = 0.5 * norm_sq;
    }
}

/// One line of Equation 2. `usage` decays for every cluster, and only the
/// clusters a batch touched get a new observation folded in. Untouched clusters
/// therefore keep their position and lose mass, which is exactly the EMA.
pub(super) fn apply_batch(
    centroids: &mut [f64],
    usage: &mut [f64],
    dimensions: usize,
    counts: &[f64],
    sums: &[f64],
    touched: &[usize],
    alpha: f64,
) {
    let decay = 1.0 - alpha;
    for mass in usage.iter_mut() {
        *mass *= decay;
    }

    for &cluster in touched {
        let count = counts[cluster];
        if count <= 0.0 {
            continue;
        }
        let mass = usage[cluster] + alpha * count;
        if mass <= 0.0 {
            continue;
        }
        let move_fraction = alpha * count / mass;
        let base = cluster * dimensions;
        for (sum, position) in sums[base..base + dimensions]
            .iter()
            .zip(centroids[base..base + dimensions].iter_mut())
        {
            let mean = *sum / count;
            *position += move_fraction * (mean - *position);
        }
        usage[cluster] = mass;
    }
}

/// Split a donor cluster into two slightly separated children, giving one to
/// the under-used slot. The perturbation alternates sign by component, as in
/// Equation 8, so the split plane passes through the donor centroid.
pub(super) fn split_centroid(
    centroids: &mut [f64],
    dimensions: usize,
    donor: usize,
    target: usize,
) {
    let donor_base = donor * dimensions;
    let target_base = target * dimensions;
    let mut child = [0.0f64; 16];

    // Dimensions above the stack buffer are far outside the packed paths and
    // rare for e-means, so fall back to a heap row rather than a fixed limit.
    let mut heap;
    let child: &mut [f64] = if dimensions <= child.len() {
        &mut child[..dimensions]
    } else {
        heap = vec![0.0; dimensions];
        &mut heap
    };

    for part in 0..dimensions {
        let sign = if part.is_multiple_of(2) { 1.0 } else { -1.0 };
        let offset = SPLIT_EPSILON * sign;
        let value = centroids[donor_base + part];
        centroids[donor_base + part] = value * (1.0 - offset);
        child[part] = value * (1.0 + offset);
    }

    // A zero or otherwise symmetric coordinate can survive the perturbation
    // unchanged. The children must still be distinct, or the revived cluster is
    // an exact duplicate of its donor.
    if (0..dimensions).all(|part| centroids[donor_base + part] == child[part]) {
        let next = child[0].next_up();
        child[0] = if next.is_finite() {
            next
        } else {
            child[0].next_down()
        };
    }

    centroids[target_base..target_base + dimensions].copy_from_slice(child);
}

const SPLIT_EPSILON: f64 = 0.000_1;

/// Revive under-used clusters, measured in batch updates.
pub(super) fn revive_unused(
    centroids: &mut [f64],
    usage: &mut [f64],
    dimensions: usize,
    threshold: f64,
    random: &mut Scramble,
) -> usize {
    if threshold <= 0.0 || usage.is_empty() {
        return 0;
    }

    let mean = usage.iter().sum::<f64>() / usage.len() as f64;
    let cutoff = threshold * mean;
    if !usage.iter().any(|mass| *mass < cutoff) {
        return 0;
    }

    // Weighted sampling without replacement, drawn by inverse CDF over the
    // remaining donor mass. This is the same distribution as the
    // Efraimidis-Spirakis `log(u) / weight` keys, but it needs one draw per
    // split rather than one per cluster, and it avoids sorting a key per
    // cluster. A donor chosen for one under-used slot cannot be chosen again,
    // and under-used targets cannot serve as donors.
    let unused: Vec<usize> = usage
        .iter()
        .enumerate()
        .filter_map(|(cluster, mass)| (*mass < cutoff).then_some(cluster))
        .collect();
    let mut blocked = vec![false; usage.len()];
    for &cluster in &unused {
        blocked[cluster] = true;
    }

    let mut splits = 0;
    for &target in &unused {
        let total = donor_sum(usage, &blocked);
        if total <= 0.0 {
            break;
        }
        let pick = random.next() * total;
        let Some(donor) = donor_pick(usage, &blocked, pick) else {
            break;
        };

        split_centroid(centroids, dimensions, donor, target);
        let half = usage[donor] / 2.0;
        usage[donor] = half;
        usage[target] = half;
        blocked[donor] = true;
        splits += 1;
    }

    splits
}

fn donor_sum(usage: &[f64], blocked: &[bool]) -> f64 {
    usage
        .iter()
        .zip(blocked.iter())
        .filter_map(|(mass, is_blocked)| (!*is_blocked && *mass > 0.0).then_some(*mass))
        .sum()
}

fn donor_pick(usage: &[f64], blocked: &[bool], pick: f64) -> Option<usize> {
    let mut cumulative = 0.0;
    usage
        .iter()
        .zip(blocked.iter())
        .enumerate()
        .filter(|(_, (mass, is_blocked))| !**is_blocked && **mass > 0.0)
        .find_map(|(cluster, (mass, _))| {
            cumulative += *mass;
            (cumulative >= pick).then_some(cluster)
        })
}
