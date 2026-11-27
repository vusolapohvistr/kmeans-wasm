//! Assignment phase: nearest-centroid scans for the moving centroids used by
//! e-means.

use fearless_simd::{Select, Simd, SimdBase, f32x4, f64x2, u32x4, u64x2};

#[inline(always)]
fn reduce_min_f64x2<S: Simd>(best_dist: f64x2<S>, best_idx: u64x2<S>) -> usize {
    let d0 = best_dist[0];
    let d1 = best_dist[1];
    let i0 = best_idx[0];
    let i1 = best_idx[1];
    if d1 < d0 || (d1 == d0 && i1 < i0) {
        i1 as usize
    } else {
        i0 as usize
    }
}

#[inline(always)]
pub(crate) fn reduce_min_f32x4<S: Simd>(best_dist: f32x4<S>, best_idx: u32x4<S>) -> usize {
    let mut min_d = best_dist[0];
    let mut min_i = best_idx[0];
    for lane in 1..4 {
        let d = best_dist[lane];
        let i = best_idx[lane];
        if d < min_d || (d == min_d && i < min_i) {
            min_d = d;
            min_i = i;
        }
    }
    min_i as usize
}

/// 2-cluster-at-a-time `f64x2` panel scan for a single `f64` point.
pub(super) fn nearest_f64_panel<S: Simd>(
    simd: S,
    point: &[f64],
    panels: &[f64],
    dimensions: usize,
    k_pairs: usize,
) -> usize {
    let point = &point[..dimensions];
    let mut best_dist = f64x2::<S>::splat(simd, f64::INFINITY);
    let mut best_idx = u64x2::<S>::splat(simd, 0);
    let mut cur_idx = u64x2::<S>::from_slice(simd, &[0, 1]);
    let step = u64x2::<S>::splat(simd, 2);

    let panel_stride = dimensions * 2;
    for pair in 0..k_pairs {
        let base = pair * panel_stride;
        let mut accumulator = f64x2::<S>::splat(simd, 0.0);
        for (part, &p_val) in point.iter().enumerate() {
            let offset = base + part * 2;
            let centroid = f64x2::<S>::from_slice(simd, &panels[offset..offset + 2]);
            let difference = f64x2::<S>::splat(simd, p_val) - centroid;
            accumulator += difference * difference;
        }

        let mask = accumulator.simd_lt(best_dist);
        best_dist = mask.select(accumulator, best_dist);
        best_idx = mask.select(cur_idx, best_idx);
        cur_idx += step;
    }

    reduce_min_f64x2(best_dist, best_idx)
}

/// 4-point register-blocked `f64x2` panel scan for arbitrary dimensions.
pub(super) fn nearest_4x_f64_generic<S: Simd>(
    simd: S,
    points: [&[f64]; 4],
    panels: &[f64],
    dimensions: usize,
    k_pairs: usize,
) -> [usize; 4] {
    let inf = f64x2::<S>::splat(simd, f64::INFINITY);
    let zero = u64x2::<S>::splat(simd, 0);
    let mut b0 = inf;
    let mut b1 = inf;
    let mut b2 = inf;
    let mut b3 = inf;
    let mut i0 = zero;
    let mut i1 = zero;
    let mut i2 = zero;
    let mut i3 = zero;
    let mut cur_idx = u64x2::<S>::from_slice(simd, &[0, 1]);
    let step = u64x2::<S>::splat(simd, 2);

    let p0 = &points[0][..dimensions];
    let p1 = &points[1][..dimensions];
    let p2 = &points[2][..dimensions];
    let p3 = &points[3][..dimensions];
    let panel_stride = dimensions * 2;

    for pair in 0..k_pairs {
        let base = pair * panel_stride;
        let z = f64x2::<S>::splat(simd, 0.0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        for part in 0..dimensions {
            let offset = base + part * 2;
            let c = f64x2::<S>::from_slice(simd, &panels[offset..offset + 2]);
            let d0 = f64x2::<S>::splat(simd, p0[part]) - c;
            let d1 = f64x2::<S>::splat(simd, p1[part]) - c;
            let d2 = f64x2::<S>::splat(simd, p2[part]) - c;
            let d3 = f64x2::<S>::splat(simd, p3[part]) - c;
            a0 += d0 * d0;
            a1 += d1 * d1;
            a2 += d2 * d2;
            a3 += d3 * d3;
        }

        let m0 = a0.simd_lt(b0);
        b0 = m0.select(a0, b0);
        i0 = m0.select(cur_idx, i0);

        let m1 = a1.simd_lt(b1);
        b1 = m1.select(a1, b1);
        i1 = m1.select(cur_idx, i1);

        let m2 = a2.simd_lt(b2);
        b2 = m2.select(a2, b2);
        i2 = m2.select(cur_idx, i2);

        let m3 = a3.simd_lt(b3);
        b3 = m3.select(a3, b3);
        i3 = m3.select(cur_idx, i3);

        cur_idx += step;
    }

    [
        reduce_min_f64x2(b0, i0),
        reduce_min_f64x2(b1, i1),
        reduce_min_f64x2(b2, i2),
        reduce_min_f64x2(b3, i3),
    ]
}

/// Single-point algebraic `half_norm_sq - <p, c>` nearest centroid for 3D panels (`stride = 16`).
pub(super) fn nearest_f32_3_dot<S: Simd>(
    simd: S,
    point: &[f32],
    panels: &[f32],
    k_panels: usize,
) -> usize {
    let px = f32x4::<S>::splat(simd, point[0]);
    let py = f32x4::<S>::splat(simd, point[1]);
    let pz = f32x4::<S>::splat(simd, point[2]);

    let mut best_score = f32x4::<S>::splat(simd, f32::INFINITY);
    let mut best_idx = u32x4::<S>::splat(simd, 0);
    let mut cur_idx = u32x4::<S>::from_slice(simd, &[0, 1, 2, 3]);
    let step = u32x4::<S>::splat(simd, 4);

    for panel in 0..k_panels {
        let base = panel * 16;
        let cx = f32x4::<S>::from_slice(simd, &panels[base..base + 4]);
        let cy = f32x4::<S>::from_slice(simd, &panels[base + 4..base + 8]);
        let cz = f32x4::<S>::from_slice(simd, &panels[base + 8..base + 12]);
        let hns = f32x4::<S>::from_slice(simd, &panels[base + 12..base + 16]);
        let score = hns - (px * cx + py * cy + pz * cz);
        let mask = score.simd_lt(best_score);
        best_score = mask.select(score, best_score);
        best_idx = mask.select(cur_idx, best_idx);
        cur_idx += step;
    }

    reduce_min_f32x4(best_score, best_idx)
}

/// Single-point algebraic `half_norm_sq - <p, c>` nearest centroid for 4D panels (`stride = 20`).
pub(super) fn nearest_f32_4_dot<S: Simd>(
    simd: S,
    point: &[f32],
    panels: &[f32],
    k_panels: usize,
) -> usize {
    let px = f32x4::<S>::splat(simd, point[0]);
    let py = f32x4::<S>::splat(simd, point[1]);
    let pz = f32x4::<S>::splat(simd, point[2]);
    let pw = f32x4::<S>::splat(simd, point[3]);

    let mut best_score = f32x4::<S>::splat(simd, f32::INFINITY);
    let mut best_idx = u32x4::<S>::splat(simd, 0);
    let mut cur_idx = u32x4::<S>::from_slice(simd, &[0, 1, 2, 3]);
    let step = u32x4::<S>::splat(simd, 4);

    for panel in 0..k_panels {
        let base = panel * 20;
        let cx = f32x4::<S>::from_slice(simd, &panels[base..base + 4]);
        let cy = f32x4::<S>::from_slice(simd, &panels[base + 4..base + 8]);
        let cz = f32x4::<S>::from_slice(simd, &panels[base + 8..base + 12]);
        let cw = f32x4::<S>::from_slice(simd, &panels[base + 12..base + 16]);
        let hns = f32x4::<S>::from_slice(simd, &panels[base + 16..base + 20]);
        let score = hns - (px * cx + py * cy + pz * cz + pw * cw);
        let mask = score.simd_lt(best_score);
        best_score = mask.select(score, best_score);
        best_idx = mask.select(cur_idx, best_idx);
        cur_idx += step;
    }

    reduce_min_f32x4(best_score, best_idx)
}

/// 4-point register-blocked `half_norm_sq - <p, c>` nearest centroid for 3D panels (`stride = 16`).
pub(super) fn nearest_4x_f32_3_dot<S: Simd>(
    simd: S,
    p: [[f32; 3]; 4],
    panels: &[f32],
    k_panels: usize,
) -> [usize; 4] {
    let p0x = f32x4::<S>::splat(simd, p[0][0]);
    let p0y = f32x4::<S>::splat(simd, p[0][1]);
    let p0z = f32x4::<S>::splat(simd, p[0][2]);
    let p1x = f32x4::<S>::splat(simd, p[1][0]);
    let p1y = f32x4::<S>::splat(simd, p[1][1]);
    let p1z = f32x4::<S>::splat(simd, p[1][2]);
    let p2x = f32x4::<S>::splat(simd, p[2][0]);
    let p2y = f32x4::<S>::splat(simd, p[2][1]);
    let p2z = f32x4::<S>::splat(simd, p[2][2]);
    let p3x = f32x4::<S>::splat(simd, p[3][0]);
    let p3y = f32x4::<S>::splat(simd, p[3][1]);
    let p3z = f32x4::<S>::splat(simd, p[3][2]);

    let inf = f32x4::<S>::splat(simd, f32::INFINITY);
    let zero = u32x4::<S>::splat(simd, 0);
    let mut b0 = inf;
    let mut b1 = inf;
    let mut b2 = inf;
    let mut b3 = inf;
    let mut i0 = zero;
    let mut i1 = zero;
    let mut i2 = zero;
    let mut i3 = zero;
    let mut cur_idx = u32x4::<S>::from_slice(simd, &[0, 1, 2, 3]);
    let step = u32x4::<S>::splat(simd, 4);

    for panel in 0..k_panels {
        let base = panel * 16;
        let cx = f32x4::<S>::from_slice(simd, &panels[base..base + 4]);
        let cy = f32x4::<S>::from_slice(simd, &panels[base + 4..base + 8]);
        let cz = f32x4::<S>::from_slice(simd, &panels[base + 8..base + 12]);
        let hns = f32x4::<S>::from_slice(simd, &panels[base + 12..base + 16]);

        let s0 = hns - (p0x * cx + p0y * cy + p0z * cz);
        let m0 = s0.simd_lt(b0);
        b0 = m0.select(s0, b0);
        i0 = m0.select(cur_idx, i0);

        let s1 = hns - (p1x * cx + p1y * cy + p1z * cz);
        let m1 = s1.simd_lt(b1);
        b1 = m1.select(s1, b1);
        i1 = m1.select(cur_idx, i1);

        let s2 = hns - (p2x * cx + p2y * cy + p2z * cz);
        let m2 = s2.simd_lt(b2);
        b2 = m2.select(s2, b2);
        i2 = m2.select(cur_idx, i2);

        let s3 = hns - (p3x * cx + p3y * cy + p3z * cz);
        let m3 = s3.simd_lt(b3);
        b3 = m3.select(s3, b3);
        i3 = m3.select(cur_idx, i3);

        cur_idx += step;
    }

    [
        reduce_min_f32x4(b0, i0),
        reduce_min_f32x4(b1, i1),
        reduce_min_f32x4(b2, i2),
        reduce_min_f32x4(b3, i3),
    ]
}

/// 4-point register-blocked `half_norm_sq - <p, c>` nearest centroid for 4D panels (`stride = 20`).
pub(super) fn nearest_4x_f32_4_dot<S: Simd>(
    simd: S,
    p: [[f32; 4]; 4],
    panels: &[f32],
    k_panels: usize,
) -> [usize; 4] {
    let p0x = f32x4::<S>::splat(simd, p[0][0]);
    let p0y = f32x4::<S>::splat(simd, p[0][1]);
    let p0z = f32x4::<S>::splat(simd, p[0][2]);
    let p0w = f32x4::<S>::splat(simd, p[0][3]);
    let p1x = f32x4::<S>::splat(simd, p[1][0]);
    let p1y = f32x4::<S>::splat(simd, p[1][1]);
    let p1z = f32x4::<S>::splat(simd, p[1][2]);
    let p1w = f32x4::<S>::splat(simd, p[1][3]);
    let p2x = f32x4::<S>::splat(simd, p[2][0]);
    let p2y = f32x4::<S>::splat(simd, p[2][1]);
    let p2z = f32x4::<S>::splat(simd, p[2][2]);
    let p2w = f32x4::<S>::splat(simd, p[2][3]);
    let p3x = f32x4::<S>::splat(simd, p[3][0]);
    let p3y = f32x4::<S>::splat(simd, p[3][1]);
    let p3z = f32x4::<S>::splat(simd, p[3][2]);
    let p3w = f32x4::<S>::splat(simd, p[3][3]);

    let inf = f32x4::<S>::splat(simd, f32::INFINITY);
    let zero = u32x4::<S>::splat(simd, 0);
    let mut b0 = inf;
    let mut b1 = inf;
    let mut b2 = inf;
    let mut b3 = inf;
    let mut i0 = zero;
    let mut i1 = zero;
    let mut i2 = zero;
    let mut i3 = zero;
    let mut cur_idx = u32x4::<S>::from_slice(simd, &[0, 1, 2, 3]);
    let step = u32x4::<S>::splat(simd, 4);

    for panel in 0..k_panels {
        let base = panel * 20;
        let cx = f32x4::<S>::from_slice(simd, &panels[base..base + 4]);
        let cy = f32x4::<S>::from_slice(simd, &panels[base + 4..base + 8]);
        let cz = f32x4::<S>::from_slice(simd, &panels[base + 8..base + 12]);
        let cw = f32x4::<S>::from_slice(simd, &panels[base + 12..base + 16]);
        let hns = f32x4::<S>::from_slice(simd, &panels[base + 16..base + 20]);

        let s0 = hns - (p0x * cx + p0y * cy + p0z * cz + p0w * cw);
        let m0 = s0.simd_lt(b0);
        b0 = m0.select(s0, b0);
        i0 = m0.select(cur_idx, i0);

        let s1 = hns - (p1x * cx + p1y * cy + p1z * cz + p1w * cw);
        let m1 = s1.simd_lt(b1);
        b1 = m1.select(s1, b1);
        i1 = m1.select(cur_idx, i1);

        let s2 = hns - (p2x * cx + p2y * cy + p2z * cz + p2w * cw);
        let m2 = s2.simd_lt(b2);
        b2 = m2.select(s2, b2);
        i2 = m2.select(cur_idx, i2);

        let s3 = hns - (p3x * cx + p3y * cy + p3z * cz + p3w * cw);
        let m3 = s3.simd_lt(b3);
        b3 = m3.select(s3, b3);
        i3 = m3.select(cur_idx, i3);

        cur_idx += step;
    }

    [
        reduce_min_f32x4(b0, i0),
        reduce_min_f32x4(b1, i1),
        reduce_min_f32x4(b2, i2),
        reduce_min_f32x4(b3, i3),
    ]
}

/// Branchless 4-cluster-at-a-time `f32x4` panel scan for arbitrary dimensions.
pub(super) fn nearest_centroid_f32_generic<S: Simd>(
    simd: S,
    point: &[f32],
    panels: &[f32],
    dimensions: usize,
    k_panels: usize,
) -> usize {
    let mut best_dist = f32x4::<S>::splat(simd, f32::INFINITY);
    let mut best_idx = u32x4::<S>::splat(simd, 0);
    let mut cur_idx = u32x4::<S>::from_slice(simd, &[0, 1, 2, 3]);
    let step = u32x4::<S>::splat(simd, 4);

    let panel_stride = dimensions * 4;
    for panel in 0..k_panels {
        let base = panel * panel_stride;
        let mut acc = f32x4::<S>::splat(simd, 0.0);
        for (part, &p_val) in point[..dimensions].iter().enumerate() {
            let offset = base + part * 4;
            let c = f32x4::<S>::from_slice(simd, &panels[offset..offset + 4]);
            let d = f32x4::<S>::splat(simd, p_val) - c;
            acc += d * d;
        }
        let mask = acc.simd_lt(best_dist);
        best_dist = mask.select(acc, best_dist);
        best_idx = mask.select(cur_idx, best_idx);
        cur_idx += step;
    }

    reduce_min_f32x4(best_dist, best_idx)
}

/// 4-point register-blocked `(p - c)^2` `f32x4` panel scan for arbitrary dimensions.
pub(super) fn nearest_4x_f32_generic<S: Simd>(
    simd: S,
    points: [&[f32]; 4],
    panels: &[f32],
    dimensions: usize,
    k_panels: usize,
) -> [usize; 4] {
    let inf = f32x4::<S>::splat(simd, f32::INFINITY);
    let zero = u32x4::<S>::splat(simd, 0);
    let mut b0 = inf;
    let mut b1 = inf;
    let mut b2 = inf;
    let mut b3 = inf;
    let mut i0 = zero;
    let mut i1 = zero;
    let mut i2 = zero;
    let mut i3 = zero;
    let mut cur_idx = u32x4::<S>::from_slice(simd, &[0, 1, 2, 3]);
    let step = u32x4::<S>::splat(simd, 4);

    let p0 = &points[0][..dimensions];
    let p1 = &points[1][..dimensions];
    let p2 = &points[2][..dimensions];
    let p3 = &points[3][..dimensions];
    let panel_stride = dimensions * 4;

    for panel in 0..k_panels {
        let base = panel * panel_stride;
        let z = f32x4::<S>::splat(simd, 0.0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        for part in 0..dimensions {
            let offset = base + part * 4;
            let c = f32x4::<S>::from_slice(simd, &panels[offset..offset + 4]);
            let d0 = f32x4::<S>::splat(simd, p0[part]) - c;
            let d1 = f32x4::<S>::splat(simd, p1[part]) - c;
            let d2 = f32x4::<S>::splat(simd, p2[part]) - c;
            let d3 = f32x4::<S>::splat(simd, p3[part]) - c;
            a0 += d0 * d0;
            a1 += d1 * d1;
            a2 += d2 * d2;
            a3 += d3 * d3;
        }

        let m0 = a0.simd_lt(b0);
        b0 = m0.select(a0, b0);
        i0 = m0.select(cur_idx, i0);

        let m1 = a1.simd_lt(b1);
        b1 = m1.select(a1, b1);
        i1 = m1.select(cur_idx, i1);

        let m2 = a2.simd_lt(b2);
        b2 = m2.select(a2, b2);
        i2 = m2.select(cur_idx, i2);

        let m3 = a3.simd_lt(b3);
        b3 = m3.select(a3, b3);
        i3 = m3.select(cur_idx, i3);

        cur_idx += step;
    }

    [
        reduce_min_f32x4(b0, i0),
        reduce_min_f32x4(b1, i1),
        reduce_min_f32x4(b2, i2),
        reduce_min_f32x4(b3, i3),
    ]
}
