//! Update phase: Hamerly bounds updates and centroid moves.

/// The optional inputs, which are absent on the production path.
///
/// Bundled rather than passed separately so the core keeps a readable argument
/// list and so it is obvious at a glance that neither is set in normal use.
#[derive(Clone, Copy, Default)]
pub struct Optional<'a> {
    /// A weight per point, for input where many points are exactly equal.
    ///
    /// Equal points always take the same cluster and contribute the same amount
    /// to that cluster's sum, so one weighted entry standing for `w` equal points
    /// gives the same clustering as `w` copies of it. Must be `None` or the same
    /// length as `points` divided by `dimensions`.
    pub weights: Option<&'a [f64]>,

    /// Replaces the random draw, for comparing two forms from one start.
    ///
    /// This exists so a difference in the algorithm can be told apart from two
    /// different local optima, which is otherwise impossible to do from outside.
    pub initial_centroids: Option<&'a [f64]>,
}

pub(super) fn update_bounds(
    centroid_distance_to_previous_position: &[f64], // p(j) – distance that c(j) last moved
    point_states: &mut [super::initialize::PointState],
) {
    let r = centroid_distance_to_previous_position
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let r_another = centroid_distance_to_previous_position
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != r)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;

    #[cfg(feature = "counters")]
    crate::work::count(&crate::work::BOUND_UPDATES, point_states.len() as u64);

    for state in point_states.iter_mut() {
        state.upper_bound += centroid_distance_to_previous_position[state.centroid as usize];
        if state.centroid as usize == r {
            state.lower_bound -= centroid_distance_to_previous_position[r_another];
        } else {
            state.lower_bound -= centroid_distance_to_previous_position[r];
        }
    }
}

pub(super) fn move_centers(
    centroids: &mut [f64],
    centroid_points_sum: &[f64],
    centroid_points_counts: &[f64],
    dimensions: usize,
    k: usize,
    centroid_distance_to_previous_position: &mut [f64],
) -> f64 {
    let mut total_squared_distance_moved = 0.0;
    for j in 0..k {
        let centroid_offset = j * dimensions;
        let points_count = centroid_points_counts[j];

        // An empty cluster has no mean to move to, so dividing by its zero count
        // produces a NaN centroid, and NaN saturates to 0 in the packed output, so
        // the caller receives a spurious black palette entry.
        //
        // Seeds are distinct values, so while k is at or below the number of
        // distinct values this cannot happen: every centroid sits on an input
        // point, that point is at distance zero, and no other centroid is, so it
        // always keeps its cluster. An empty cluster is only reachable when k
        // exceeds the number of distinct values, which is forced rather than
        // unlucky. Recording no movement keeps the bounds finite.
        if points_count == 0.0 {
            centroid_distance_to_previous_position[j] = 0.0;
            continue;
        }

        let mut squared_distance_moved = 0.0;
        for part in 0..dimensions {
            let index = centroid_offset + part;
            let previous_position = centroids[index];
            let moved = centroid_points_sum[index] / points_count;
            centroids[index] = moved;
            let difference = previous_position - moved;
            squared_distance_moved += difference * difference;
        }
        centroid_distance_to_previous_position[j] = squared_distance_moved.sqrt();
        total_squared_distance_moved += squared_distance_moved;
    }
    total_squared_distance_moved
}
