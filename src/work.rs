//! Deterministic work counters.
//!
//! Wall clock on a shared machine is not trustworthy, and the question these
//! answer, "how much of the work is the distance pass and how much is the bound
//! bookkeeping", is exactly the kind that counters settle and a stopwatch
//! cannot. Everything here is a count, so it is exact and reproducible.
//!
//! The counters sit in the innermost loop, so they are behind a feature and
//! compiled out of the shipped build entirely. Anything timed with them on is
//! meaningless: use them for counts, and take timings from a separate run.

use std::sync::atomic::{AtomicU64, Ordering};

/// One squared distance between a point and a centroid. This is the work a GPU
/// could take, and it is order-independent, so it can be computed anywhere and
/// still be bit-identical.
pub static DISTANCE_EVALUATIONS: AtomicU64 = AtomicU64::new(0);

/// One point reaching the per-point bound test. Pure arithmetic on two numbers
/// already in registers, and it decides whether any distance is computed at all,
/// so it has to stay where the loop-carried state is.
pub static BOUND_CHECKS: AtomicU64 = AtomicU64::new(0);

/// Points pushed past the bound test into a full scan of every centroid. This is
/// the expensive branch and it is the one that decides whether a round costs `n`
/// distances or `n * k`.
pub static FULL_SCANS: AtomicU64 = AtomicU64::new(0);

/// One point having its two bounds moved by the centroid movements. Sequential
/// across points and cheap, but it touches every point every round, so at a
/// hundred rounds it is not free.
pub static BOUND_UPDATES: AtomicU64 = AtomicU64::new(0);

/// One point's contribution added to or removed from a running cluster sum.
/// Order-sensitive, so it can never move to a GPU, and it is the reason the
/// centroids stay bit-identical.
pub static SUM_UPDATES: AtomicU64 = AtomicU64::new(0);

/// Everything above, in one read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub distance_evaluations: u64,
    pub bound_checks: u64,
    pub full_scans: u64,
    pub bound_updates: u64,
    pub sum_updates: u64,
}

pub fn reset() {
    for counter in [
        &DISTANCE_EVALUATIONS,
        &BOUND_CHECKS,
        &FULL_SCANS,
        &BOUND_UPDATES,
        &SUM_UPDATES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

pub fn snapshot() -> Counts {
    Counts {
        distance_evaluations: DISTANCE_EVALUATIONS.load(Ordering::Relaxed),
        bound_checks: BOUND_CHECKS.load(Ordering::Relaxed),
        full_scans: FULL_SCANS.load(Ordering::Relaxed),
        bound_updates: BOUND_UPDATES.load(Ordering::Relaxed),
        sum_updates: SUM_UPDATES.load(Ordering::Relaxed),
    }
}

/// The counters are relaxed because they are statistics, not synchronisation:
/// nothing branches on them, and a run is single-threaded.
#[inline(always)]
pub fn count(counter: &AtomicU64, by: u64) {
    counter.fetch_add(by, Ordering::Relaxed);
}

#[inline(always)]
pub fn count_one(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}
