//! Small shared utilities: the local PRNG stream and the shuffle helper.

use crate::kmeans_triangle::Random;

/// The private stream used after the initial centroids are drawn.
///
/// SplitMix64 is enough here: this is a shuffle, not a simulation, and the only
/// requirement is a uniform-enough permutation from a seed the caller can
/// reproduce. Using it avoids one call into JavaScript per shuffled item on the
/// wasm target, which is a large fixed cost on a million-point pass.
pub(super) struct Scramble(u64);

impl Scramble {
    pub(super) fn from_random(random: &mut Random) -> Self {
        let first = random.next().to_bits();
        let second = random.next().to_bits();
        Self(first ^ second.rotate_left(32) ^ 0x9e37_79b9_7f4a_7c15)
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^ (mixed >> 31)
    }

    #[inline(always)]
    pub(super) fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub(super) fn next(&mut self) -> f64 {
        // 53 significant bits, so every representable value in [0, 1) is
        // reachable and the conversion is exact.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

pub(super) fn shuffle<T>(items: &mut [T], random: &mut Scramble) {
    for index in (1..items.len()).rev() {
        let other = ((u64::from(random.next_u32()) * (index as u64 + 1)) >> 32) as usize;
        items.swap(index, other);
    }
}
