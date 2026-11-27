//! Collapse a packed colour buffer to its distinct values with weights, and
//! map pixels onto palettes with `f32x4` SIMD + Voronoi ball hints.

use fearless_simd::{Level, Select, Simd, SimdBase, dispatch, f32x4, u32x4};

use crate::emeans::assignment::reduce_min_f32x4;

/// Slots are a power of two, so the probe sequence is `key * PHI >> shift`.
const PHI: u32 = 2_654_435_761;

/// Fixed table capacity (`2^18` slots; bail-out at 196,608 distinct values).
const MAX_SLOTS: usize = 1 << 18;

const MAX_LOAD_NUMERATOR: usize = 3;
const MAX_LOAD_DENOMINATOR: usize = 4;

/// The distinct values, widened to `f64`, with a weight per value.
pub struct Collapsed {
    pub points: Vec<f64>,
    pub weights: Vec<f64>,
    pub distinct: usize,
}

#[inline(always)]
fn key_of_fixed<const C: usize>(pixel: &[u8]) -> u32 {
    if C == 4 {
        u32::from_be_bytes([pixel[0], pixel[1], pixel[2], pixel[3]])
    } else {
        (u32::from(pixel[0]) << 16) | (u32::from(pixel[1]) << 8) | u32::from(pixel[2])
    }
}

fn round_up_to_power_of_two(value: usize) -> usize {
    let mut slots = 1;
    while slots < value {
        slots <<= 1;
    }
    slots
}

#[derive(Clone, Copy, Default)]
struct Slot {
    key: u32,
    value: u32,
}

/// Open-addressing colour table.
pub(crate) struct Table {
    entries: Vec<Slot>,
    shift: u32,
    mask: usize,
    distinct: usize,
}

impl Table {
    pub(crate) fn build(slice: &[u8], components: usize) -> Option<Self> {
        match components {
            3 => Self::build_fixed::<3>(slice),
            4 => Self::build_fixed::<4>(slice),
            _ => None,
        }
    }

    fn build_fixed<const C: usize>(slice: &[u8]) -> Option<Self> {
        let pixel_count = slice.len() / C;
        if pixel_count == 0 {
            return None;
        }
        let slots = round_up_to_power_of_two(pixel_count.saturating_mul(2).min(MAX_SLOTS));
        let load_limit = (slots / MAX_LOAD_DENOMINATOR * MAX_LOAD_NUMERATOR).max(1);
        let shift = 32 - slots.trailing_zeros();

        let mut entries = vec![Slot::default(); slots];
        let mut distinct = 0usize;
        let mask = slots - 1;
        let mut prev_key = key_of_fixed::<C>(&slice[..C]) ^ 1;
        let mut prev_index = 0usize;

        for pixel in slice.as_chunks::<C>().0 {
            let key = key_of_fixed::<C>(pixel);
            if key == prev_key {
                entries[prev_index].value += 1;
                continue;
            }

            let mut index = (key.wrapping_mul(PHI) >> shift) as usize;
            loop {
                let slot = &mut entries[index];
                if slot.value == 0 {
                    slot.key = key;
                    slot.value = 1;
                    distinct += 1;
                    if distinct > load_limit {
                        return None;
                    }
                    prev_key = key;
                    prev_index = index;
                    break;
                }
                if slot.key == key {
                    slot.value += 1;
                    prev_key = key;
                    prev_index = index;
                    break;
                }
                index = (index + 1) & mask;
            }
        }

        Some(Table {
            entries,
            shift,
            mask,
            distinct,
        })
    }
}

/// 4-lane panel-packed `f32x4` palette for bit-exact nearest searches on `u8`
/// inputs (`max_dist^2 = 260,100 < 2^24`), using algebraic `norm_sq - 2<p, c>`
/// scoring and Voronoi inscribed-ball fast paths.
///
/// Panel layout per 4 entries: `[r0..3, g0..3, b0..3, norm_sq0..3]` (`stride=16`)
/// for RGB, or `[r0..3, g0..3, b0..3, a0..3, norm_sq0..3]` (`stride=20`) for RGBA.
pub(crate) struct PaletteLut {
    k_panels: usize,
    panels: Vec<f32>,
    min_dist_sq: Vec<f32>,
}

impl PaletteLut {
    pub(crate) fn new(palette: &[u8], components: usize) -> Self {
        let entries = palette.len() / components;
        let k_panels = entries.div_ceil(4);
        let stride = if components == 4 { 20 } else { 16 };
        let mut panels = vec![f32::INFINITY; k_panels * stride];

        for (i, chunk) in palette.chunks_exact(components).enumerate() {
            let panel = (i >> 2) * stride;
            let lane = i & 3;
            let rv = f32::from(chunk[0]);
            let gv = f32::from(chunk[1]);
            let bv = f32::from(chunk[2]);
            panels[panel + lane] = rv;
            panels[panel + 4 + lane] = gv;
            panels[panel + 8 + lane] = bv;
            let mut n2 = rv * rv + gv * gv + bv * bv;
            if components == 4 {
                let av = f32::from(chunk[3]);
                panels[panel + 12 + lane] = av;
                n2 += av * av;
                panels[panel + 16 + lane] = n2;
            } else {
                panels[panel + 12 + lane] = n2;
            }
        }

        let mut min_dist_sq = vec![f32::INFINITY; entries];
        for i in 0..entries {
            let ib = (i >> 2) * stride + (i & 3);
            let ri = panels[ib];
            let gi = panels[ib + 4];
            let bi = panels[ib + 8];
            let ai = if components == 4 {
                panels[ib + 12]
            } else {
                0.0
            };
            for j in (i + 1)..entries {
                let jb = (j >> 2) * stride + (j & 3);
                let dr = ri - panels[jb];
                let dg = gi - panels[jb + 4];
                let db = bi - panels[jb + 8];
                let mut d2 = dr * dr + dg * dg + db * db;
                if components == 4 {
                    let da = ai - panels[jb + 12];
                    d2 += da * da;
                }
                if d2 < min_dist_sq[i] {
                    min_dist_sq[i] = d2;
                }
                if d2 < min_dist_sq[j] {
                    min_dist_sq[j] = d2;
                }
            }
        }

        Self {
            k_panels,
            panels,
            min_dist_sq,
        }
    }

    #[inline(always)]
    fn voronoi_hit_3(&self, p0: f32, p1: f32, p2: f32, hint: usize) -> bool {
        let base = (hint >> 2) * 16 + (hint & 3);
        let dr = p0 - self.panels[base];
        let dg = p1 - self.panels[base + 4];
        let db = p2 - self.panels[base + 8];
        let d2 = dr * dr + dg * dg + db * db;
        4.0 * d2 < self.min_dist_sq[hint]
    }

    #[inline(always)]
    fn voronoi_hit_4(&self, p0: f32, p1: f32, p2: f32, p3: f32, hint: usize) -> bool {
        let base = (hint >> 2) * 20 + (hint & 3);
        let dr = p0 - self.panels[base];
        let dg = p1 - self.panels[base + 4];
        let db = p2 - self.panels[base + 8];
        let da = p3 - self.panels[base + 12];
        let d2 = dr * dr + dg * dg + db * db + da * da;
        4.0 * d2 < self.min_dist_sq[hint]
    }

    #[inline]
    pub(crate) fn nearest_3<S: Simd>(&self, simd: S, p0: f32, p1: f32, p2: f32) -> usize {
        let pr = f32x4::<S>::splat(simd, -2.0 * p0);
        let pg = f32x4::<S>::splat(simd, -2.0 * p1);
        let pb = f32x4::<S>::splat(simd, -2.0 * p2);

        let mut best_score = f32x4::<S>::splat(simd, f32::INFINITY);
        let mut best_idx = u32x4::<S>::splat(simd, 0);
        let mut cur_idx = u32x4::<S>::from_slice(simd, &[0, 1, 2, 3]);
        let step = u32x4::<S>::splat(simd, 4);

        for panel in 0..self.k_panels {
            let base = panel * 16;
            let cr = f32x4::<S>::from_slice(simd, &self.panels[base..base + 4]);
            let cg = f32x4::<S>::from_slice(simd, &self.panels[base + 4..base + 8]);
            let cb = f32x4::<S>::from_slice(simd, &self.panels[base + 8..base + 12]);
            let n2 = f32x4::<S>::from_slice(simd, &self.panels[base + 12..base + 16]);
            let score = n2 + pr * cr + pg * cg + pb * cb;
            let mask = score.simd_lt(best_score);
            best_score = mask.select(score, best_score);
            best_idx = mask.select(cur_idx, best_idx);
            cur_idx += step;
        }

        reduce_min_f32x4(best_score, best_idx)
    }

    pub(crate) fn nearest_4x_3<S: Simd>(&self, simd: S, p: [[f32; 3]; 4]) -> [usize; 4] {
        let p0r = f32x4::<S>::splat(simd, -2.0 * p[0][0]);
        let p0g = f32x4::<S>::splat(simd, -2.0 * p[0][1]);
        let p0b = f32x4::<S>::splat(simd, -2.0 * p[0][2]);
        let p1r = f32x4::<S>::splat(simd, -2.0 * p[1][0]);
        let p1g = f32x4::<S>::splat(simd, -2.0 * p[1][1]);
        let p1b = f32x4::<S>::splat(simd, -2.0 * p[1][2]);
        let p2r = f32x4::<S>::splat(simd, -2.0 * p[2][0]);
        let p2g = f32x4::<S>::splat(simd, -2.0 * p[2][1]);
        let p2b = f32x4::<S>::splat(simd, -2.0 * p[2][2]);
        let p3r = f32x4::<S>::splat(simd, -2.0 * p[3][0]);
        let p3g = f32x4::<S>::splat(simd, -2.0 * p[3][1]);
        let p3b = f32x4::<S>::splat(simd, -2.0 * p[3][2]);

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

        for panel in 0..self.k_panels {
            let base = panel * 16;
            let cr = f32x4::<S>::from_slice(simd, &self.panels[base..base + 4]);
            let cg = f32x4::<S>::from_slice(simd, &self.panels[base + 4..base + 8]);
            let cb = f32x4::<S>::from_slice(simd, &self.panels[base + 8..base + 12]);
            let n2 = f32x4::<S>::from_slice(simd, &self.panels[base + 12..base + 16]);

            let s0 = n2 + p0r * cr + p0g * cg + p0b * cb;
            let m0 = s0.simd_lt(b0);
            b0 = m0.select(s0, b0);
            i0 = m0.select(cur_idx, i0);

            let s1 = n2 + p1r * cr + p1g * cg + p1b * cb;
            let m1 = s1.simd_lt(b1);
            b1 = m1.select(s1, b1);
            i1 = m1.select(cur_idx, i1);

            let s2 = n2 + p2r * cr + p2g * cg + p2b * cb;
            let m2 = s2.simd_lt(b2);
            b2 = m2.select(s2, b2);
            i2 = m2.select(cur_idx, i2);

            let s3 = n2 + p3r * cr + p3g * cg + p3b * cb;
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

    #[inline(always)]
    pub(crate) fn nearest_3_with_hint<S: Simd>(
        &self,
        simd: S,
        p0: f32,
        p1: f32,
        p2: f32,
        hint: usize,
    ) -> usize {
        if self.voronoi_hit_3(p0, p1, p2, hint) {
            return hint;
        }
        self.nearest_3(simd, p0, p1, p2)
    }

    #[inline]
    pub(crate) fn nearest_4<S: Simd>(&self, simd: S, p0: f32, p1: f32, p2: f32, p3: f32) -> usize {
        let pr = f32x4::<S>::splat(simd, -2.0 * p0);
        let pg = f32x4::<S>::splat(simd, -2.0 * p1);
        let pb = f32x4::<S>::splat(simd, -2.0 * p2);
        let pa = f32x4::<S>::splat(simd, -2.0 * p3);

        let mut best_score = f32x4::<S>::splat(simd, f32::INFINITY);
        let mut best_idx = u32x4::<S>::splat(simd, 0);
        let mut cur_idx = u32x4::<S>::from_slice(simd, &[0, 1, 2, 3]);
        let step = u32x4::<S>::splat(simd, 4);

        for panel in 0..self.k_panels {
            let base = panel * 20;
            let cr = f32x4::<S>::from_slice(simd, &self.panels[base..base + 4]);
            let cg = f32x4::<S>::from_slice(simd, &self.panels[base + 4..base + 8]);
            let cb = f32x4::<S>::from_slice(simd, &self.panels[base + 8..base + 12]);
            let ca = f32x4::<S>::from_slice(simd, &self.panels[base + 12..base + 16]);
            let n2 = f32x4::<S>::from_slice(simd, &self.panels[base + 16..base + 20]);
            let score = n2 + pr * cr + pg * cg + pb * cb + pa * ca;
            let mask = score.simd_lt(best_score);
            best_score = mask.select(score, best_score);
            best_idx = mask.select(cur_idx, best_idx);
            cur_idx += step;
        }

        reduce_min_f32x4(best_score, best_idx)
    }

    pub(crate) fn nearest_4x_4<S: Simd>(&self, simd: S, p: [[f32; 4]; 4]) -> [usize; 4] {
        let p0r = f32x4::<S>::splat(simd, -2.0 * p[0][0]);
        let p0g = f32x4::<S>::splat(simd, -2.0 * p[0][1]);
        let p0b = f32x4::<S>::splat(simd, -2.0 * p[0][2]);
        let p0a = f32x4::<S>::splat(simd, -2.0 * p[0][3]);
        let p1r = f32x4::<S>::splat(simd, -2.0 * p[1][0]);
        let p1g = f32x4::<S>::splat(simd, -2.0 * p[1][1]);
        let p1b = f32x4::<S>::splat(simd, -2.0 * p[1][2]);
        let p1a = f32x4::<S>::splat(simd, -2.0 * p[1][3]);
        let p2r = f32x4::<S>::splat(simd, -2.0 * p[2][0]);
        let p2g = f32x4::<S>::splat(simd, -2.0 * p[2][1]);
        let p2b = f32x4::<S>::splat(simd, -2.0 * p[2][2]);
        let p2a = f32x4::<S>::splat(simd, -2.0 * p[2][3]);
        let p3r = f32x4::<S>::splat(simd, -2.0 * p[3][0]);
        let p3g = f32x4::<S>::splat(simd, -2.0 * p[3][1]);
        let p3b = f32x4::<S>::splat(simd, -2.0 * p[3][2]);
        let p3a = f32x4::<S>::splat(simd, -2.0 * p[3][3]);

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

        for panel in 0..self.k_panels {
            let base = panel * 20;
            let cr = f32x4::<S>::from_slice(simd, &self.panels[base..base + 4]);
            let cg = f32x4::<S>::from_slice(simd, &self.panels[base + 4..base + 8]);
            let cb = f32x4::<S>::from_slice(simd, &self.panels[base + 8..base + 12]);
            let ca = f32x4::<S>::from_slice(simd, &self.panels[base + 12..base + 16]);
            let n2 = f32x4::<S>::from_slice(simd, &self.panels[base + 16..base + 20]);

            let s0 = n2 + p0r * cr + p0g * cg + p0b * cb + p0a * ca;
            let m0 = s0.simd_lt(b0);
            b0 = m0.select(s0, b0);
            i0 = m0.select(cur_idx, i0);

            let s1 = n2 + p1r * cr + p1g * cg + p1b * cb + p1a * ca;
            let m1 = s1.simd_lt(b1);
            b1 = m1.select(s1, b1);
            i1 = m1.select(cur_idx, i1);

            let s2 = n2 + p2r * cr + p2g * cg + p2b * cb + p2a * ca;
            let m2 = s2.simd_lt(b2);
            b2 = m2.select(s2, b2);
            i2 = m2.select(cur_idx, i2);

            let s3 = n2 + p3r * cr + p3g * cg + p3b * cb + p3a * ca;
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

    #[inline(always)]
    pub(crate) fn nearest_4_with_hint<S: Simd>(
        &self,
        simd: S,
        p0: f32,
        p1: f32,
        p2: f32,
        p3: f32,
        hint: usize,
    ) -> usize {
        if self.voronoi_hit_4(p0, p1, p2, p3, hint) {
            return hint;
        }
        self.nearest_4(simd, p0, p1, p2, p3)
    }
}

/// Collapses `slice` into distinct values with weights, or returns `None` when the
/// input repeats too little for the reduction to pay.
pub fn collapse(slice: &[u8], components: usize) -> Option<Collapsed> {
    let table = Table::build(slice, components)?;
    let distinct = table.distinct;

    let mut points = Vec::with_capacity(distinct * components);
    let mut weights = Vec::with_capacity(distinct);
    for slot in &table.entries {
        if slot.value == 0 {
            continue;
        }
        for component in (0..components).rev() {
            points.push(f64::from((slot.key >> (component * 8)) as u8));
        }
        weights.push(f64::from(slot.value));
    }

    Some(Collapsed {
        points,
        weights,
        distinct,
    })
}

#[inline(always)]
fn unpack_rgb(key: u32) -> [f32; 3] {
    [
        f32::from((key >> 16) as u8),
        f32::from((key >> 8) as u8),
        f32::from(key as u8),
    ]
}

#[inline(always)]
fn unpack_rgba(key: u32) -> [f32; 4] {
    [
        f32::from((key >> 24) as u8),
        f32::from((key >> 16) as u8),
        f32::from((key >> 8) as u8),
        f32::from(key as u8),
    ]
}

/// A colour to the palette entry it becomes, resolved once per distinct colour.
pub(crate) struct PaletteMap {
    table: Table,
}

impl PaletteMap {
    pub(crate) fn build(slice: &[u8], components: usize, palette: &[u8]) -> Option<Self> {
        if palette.is_empty() {
            return None;
        }
        let mut table = Table::build(slice, components)?;
        let lut = PaletteLut::new(palette, components);
        let level = Level::baseline();

        dispatch!(level, simd => {
            let mut buf = [0usize; 4];
            let mut buf_len = 0usize;
            if components == 3 {
                for slot_idx in 0..table.entries.len() {
                    if table.entries[slot_idx].value != 0 {
                        buf[buf_len] = slot_idx;
                        buf_len += 1;
                        if buf_len == 4 {
                            let pts = [
                                unpack_rgb(table.entries[buf[0]].key),
                                unpack_rgb(table.entries[buf[1]].key),
                                unpack_rgb(table.entries[buf[2]].key),
                                unpack_rgb(table.entries[buf[3]].key),
                            ];
                            let res = lut.nearest_4x_3(simd, pts);
                            for m in 0..4 {
                                table.entries[buf[m]].value = res[m] as u32 + 1;
                            }
                            buf_len = 0;
                        }
                    }
                }
                for &slot_idx in &buf[..buf_len] {
                    let [p0, p1, p2] = unpack_rgb(table.entries[slot_idx].key);
                    table.entries[slot_idx].value = lut.nearest_3(simd, p0, p1, p2) as u32 + 1;
                }
            } else {
                for slot_idx in 0..table.entries.len() {
                    if table.entries[slot_idx].value != 0 {
                        buf[buf_len] = slot_idx;
                        buf_len += 1;
                        if buf_len == 4 {
                            let pts = [
                                unpack_rgba(table.entries[buf[0]].key),
                                unpack_rgba(table.entries[buf[1]].key),
                                unpack_rgba(table.entries[buf[2]].key),
                                unpack_rgba(table.entries[buf[3]].key),
                            ];
                            let res = lut.nearest_4x_4(simd, pts);
                            for m in 0..4 {
                                table.entries[buf[m]].value = res[m] as u32 + 1;
                            }
                            buf_len = 0;
                        }
                    }
                }
                for &slot_idx in &buf[..buf_len] {
                    let [p0, p1, p2, p3] = unpack_rgba(table.entries[slot_idx].key);
                    table.entries[slot_idx].value = lut.nearest_4(simd, p0, p1, p2, p3) as u32 + 1;
                }
            }
        });

        Some(PaletteMap { table })
    }

    #[inline(always)]
    pub(crate) fn lookup_fixed<const C: usize>(&self, pixel: &[u8]) -> usize {
        let key = key_of_fixed::<C>(pixel);
        let table = &self.table;

        let mut index = (key.wrapping_mul(PHI) >> table.shift) as usize;
        loop {
            let slot = table.entries[index];
            if slot.key == key {
                return (slot.value - 1) as usize;
            }
            if slot.value == 0 {
                return 0;
            }
            index = (index + 1) & table.mask;
        }
    }
}

/// Maps packed pixels onto an RGBA buffer ready for `putImageData`.
pub(crate) fn map_to_rgba(pixels: &[u8], palette: &[u8], components: usize) -> Vec<u8> {
    let pixel_count = pixels.len() / components;
    if pixel_count == 0 {
        return Vec::new();
    }

    let palette_rgba: Vec<[u8; 4]> = palette
        .chunks_exact(components)
        .map(|entry| {
            [
                entry[0],
                entry[1],
                entry[2],
                if components == 4 { entry[3] } else { 255 },
            ]
        })
        .collect();

    let mut output = vec![0u8; pixel_count * 4];
    let out_rows = output.as_chunks_mut::<4>().0;
    let map = PaletteMap::build(pixels, components, palette);

    match (map, components) {
        (Some(table), 4) => {
            let mut prev_key = key_of_fixed::<4>(&pixels[..4]) ^ 1;
            let mut prev_val = [0u8; 4];
            for (pixel, out) in pixels.as_chunks::<4>().0.iter().zip(out_rows.iter_mut()) {
                let key = key_of_fixed::<4>(pixel);
                if key != prev_key {
                    prev_key = key;
                    prev_val = palette_rgba[table.lookup_fixed::<4>(pixel)];
                }
                *out = prev_val;
            }
        }
        (Some(table), _) => {
            let mut prev_key = key_of_fixed::<3>(&pixels[..3]) ^ 1;
            let mut prev_val = [0u8; 4];
            for (pixel, out) in pixels.as_chunks::<3>().0.iter().zip(out_rows.iter_mut()) {
                let key = key_of_fixed::<3>(pixel);
                if key != prev_key {
                    prev_key = key;
                    prev_val = palette_rgba[table.lookup_fixed::<3>(pixel)];
                }
                *out = prev_val;
            }
        }
        (None, 4) => {
            let lut = PaletteLut::new(palette, 4);
            let level = Level::baseline();
            let mut prev = 0usize;
            dispatch!(level, simd => {
                for (pixel, out) in pixels.as_chunks::<4>().0.iter().zip(out_rows.iter_mut()) {
                    prev = lut.nearest_4_with_hint(
                        simd,
                        f32::from(pixel[0]),
                        f32::from(pixel[1]),
                        f32::from(pixel[2]),
                        f32::from(pixel[3]),
                        prev,
                    );
                    *out = palette_rgba[prev];
                }
            });
        }
        (None, _) => {
            let lut = PaletteLut::new(palette, 3);
            let level = Level::baseline();
            let mut prev = 0usize;
            dispatch!(level, simd => {
                for (pixel, out) in pixels.as_chunks::<3>().0.iter().zip(out_rows.iter_mut()) {
                    prev = lut.nearest_3_with_hint(
                        simd,
                        f32::from(pixel[0]),
                        f32::from(pixel[1]),
                        f32::from(pixel[2]),
                        prev,
                    );
                    *out = palette_rgba[prev];
                }
            });
        }
    }

    output
}

/// Maps each packed pixel to its nearest palette entry index.
pub(crate) fn map_to_indices(pixels: &[u8], palette: &[u8], components: usize) -> Vec<u32> {
    let pixel_count = pixels.len() / components;
    if pixel_count == 0 {
        return Vec::new();
    }

    let mut indices = vec![0u32; pixel_count];
    match (PaletteMap::build(pixels, components, palette), components) {
        (Some(table), 4) => {
            let mut prev_key = key_of_fixed::<4>(&pixels[..4]) ^ 1;
            let mut prev_val = 0u32;
            for (pixel, out) in pixels.as_chunks::<4>().0.iter().zip(indices.iter_mut()) {
                let key = key_of_fixed::<4>(pixel);
                if key != prev_key {
                    prev_key = key;
                    prev_val = table.lookup_fixed::<4>(pixel) as u32;
                }
                *out = prev_val;
            }
        }
        (Some(table), _) => {
            let mut prev_key = key_of_fixed::<3>(&pixels[..3]) ^ 1;
            let mut prev_val = 0u32;
            for (pixel, out) in pixels.as_chunks::<3>().0.iter().zip(indices.iter_mut()) {
                let key = key_of_fixed::<3>(pixel);
                if key != prev_key {
                    prev_key = key;
                    prev_val = table.lookup_fixed::<3>(pixel) as u32;
                }
                *out = prev_val;
            }
        }
        (None, 4) => {
            let lut = PaletteLut::new(palette, 4);
            let level = Level::baseline();
            let mut prev = 0usize;
            dispatch!(level, simd => {
                for (pixel, out) in pixels.as_chunks::<4>().0.iter().zip(indices.iter_mut()) {
                    prev = lut.nearest_4_with_hint(
                        simd,
                        f32::from(pixel[0]),
                        f32::from(pixel[1]),
                        f32::from(pixel[2]),
                        f32::from(pixel[3]),
                        prev,
                    );
                    *out = prev as u32;
                }
            });
        }
        (None, _) => {
            let lut = PaletteLut::new(palette, 3);
            let level = Level::baseline();
            let mut prev = 0usize;
            dispatch!(level, simd => {
                for (pixel, out) in pixels.as_chunks::<3>().0.iter().zip(indices.iter_mut()) {
                    prev = lut.nearest_3_with_hint(
                        simd,
                        f32::from(pixel[0]),
                        f32::from(pixel[1]),
                        f32::from(pixel[2]),
                        prev,
                    );
                    *out = prev as u32;
                }
            });
        }
    }
    indices
}

/// Maps packed pixels onto a palette with Floyd–Steinberg error diffusion,
/// returning an RGBA buffer ready for `putImageData`.
pub(crate) fn map_to_rgba_dithered(
    pixels: &[u8],
    width: usize,
    height: usize,
    palette: &[u8],
    components: usize,
) -> Vec<u8> {
    if width == 0 || height == 0 || pixels.is_empty() {
        return Vec::new();
    }

    let palette_rgba: Vec<[u8; 4]> = palette
        .chunks_exact(components)
        .map(|entry| {
            [
                entry[0],
                entry[1],
                entry[2],
                if components == 4 { entry[3] } else { 255 },
            ]
        })
        .collect();
    let palette_f32: Vec<[f32; 4]> = palette_rgba
        .iter()
        .map(|entry| {
            [
                f32::from(entry[0]),
                f32::from(entry[1]),
                f32::from(entry[2]),
                f32::from(entry[3]),
            ]
        })
        .collect();
    let lut = PaletteLut::new(palette, components);
    let mut output = vec![0u8; width * height * 4];
    let level = Level::baseline();

    dispatch!(level, simd => {
        if components == 4 {
            dither_fixed::<4, _>(simd, pixels, width, height, &lut, &palette_rgba, &palette_f32, &mut output);
        } else {
            dither_fixed::<3, _>(simd, pixels, width, height, &lut, &palette_rgba, &palette_f32, &mut output);
        }
    });

    output
}

#[allow(clippy::too_many_arguments)]
fn dither_fixed<const C: usize, S: Simd>(
    simd: S,
    pixels: &[u8],
    width: usize,
    height: usize,
    lut: &PaletteLut,
    palette_rgba: &[[u8; 4]],
    palette_f32: &[[f32; 4]],
    output: &mut [u8],
) {
    let mut cur_err = vec![[0.0f32; 4]; width + 2];
    let mut next_err = vec![[0.0f32; 4]; width + 2];
    let mut cache = vec![(0u32, 0u32); 4096];
    let out_rows = output.as_chunks_mut::<4>().0;
    let in_rows = pixels.as_chunks::<C>().0;
    let mut prev_entry = 0usize;
    let mut prev2_entry = 0usize;

    let w7 = f32x4::<S>::splat(simd, 7.0 / 16.0);
    let w3 = f32x4::<S>::splat(simd, 3.0 / 16.0);
    let w5 = f32x4::<S>::splat(simd, 5.0 / 16.0);
    let w1 = f32x4::<S>::splat(simd, 1.0 / 16.0);

    for y in 0..height {
        next_err.fill([0.0f32; 4]);
        let row_start = y * width;
        for x in 0..width {
            let px = &in_rows[row_start + x];
            let err = cur_err[x + 1];

            let t0 = (f32::from(px[0]) + err[0]).clamp(0.0, 255.0);
            let t1 = (f32::from(px[1]) + err[1]).clamp(0.0, 255.0);
            let t2 = (f32::from(px[2]) + err[2]).clamp(0.0, 255.0);
            let t3 = if C == 4 {
                (f32::from(px[3]) + err[3]).clamp(0.0, 255.0)
            } else {
                0.0
            };

            let r0 = t0.round();
            let r1 = t1.round();
            let r2 = t2.round();
            let r3 = if C == 4 { t3.round() } else { 0.0 };

            let qkey = if C == 4 {
                u32::from_be_bytes([r0 as u8, r1 as u8, r2 as u8, r3 as u8])
            } else {
                (u32::from(r0 as u8) << 16) | (u32::from(r1 as u8) << 8) | u32::from(r2 as u8)
            };

            let slot_idx = (qkey.wrapping_mul(PHI) >> 20) as usize;
            let cached = cache[slot_idx];
            let entry = if cached.1 != 0 && cached.0 == qkey {
                (cached.1 - 1) as usize
            } else {
                let resolved = if C == 4 {
                    if lut.voronoi_hit_4(r0, r1, r2, r3, prev_entry) {
                        prev_entry
                    } else if prev2_entry != prev_entry
                        && lut.voronoi_hit_4(r0, r1, r2, r3, prev2_entry)
                    {
                        prev2_entry
                    } else {
                        lut.nearest_4(simd, r0, r1, r2, r3)
                    }
                } else if lut.voronoi_hit_3(r0, r1, r2, prev_entry) {
                    prev_entry
                } else if prev2_entry != prev_entry && lut.voronoi_hit_3(r0, r1, r2, prev2_entry) {
                    prev2_entry
                } else {
                    lut.nearest_3(simd, r0, r1, r2)
                };
                cache[slot_idx] = (qkey, resolved as u32 + 1);
                resolved
            };

            if entry != prev_entry {
                prev2_entry = prev_entry;
                prev_entry = entry;
            }

            out_rows[row_start + x] = palette_rgba[entry];
            let chosen = &palette_f32[entry];
            let e = f32x4::<S>::from_slice(simd, &[t0, t1, t2, t3])
                - f32x4::<S>::from_slice(simd, chosen);

            let c_right = f32x4::<S>::from_slice(simd, &cur_err[x + 2]) + e * w7;
            let n_left = f32x4::<S>::from_slice(simd, &next_err[x]) + e * w3;
            let n_mid = f32x4::<S>::from_slice(simd, &next_err[x + 1]) + e * w5;
            let n_right = f32x4::<S>::from_slice(simd, &next_err[x + 2]) + e * w1;

            cur_err[x + 2] = [c_right[0], c_right[1], c_right[2], c_right[3]];
            next_err[x] = [n_left[0], n_left[1], n_left[2], n_left[3]];
            next_err[x + 1] = [n_mid[0], n_mid[1], n_mid[2], n_mid[3]];
            next_err[x + 2] = [n_right[0], n_right[1], n_right[2], n_right[3]];
        }
        std::mem::swap(&mut cur_err, &mut next_err);
    }
}
