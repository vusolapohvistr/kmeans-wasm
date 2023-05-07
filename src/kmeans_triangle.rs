pub struct HamerlyKmeansResult {
    pub centroids: Vec<Vec<f64>>,
    pub iterations: usize,
    pub point_centroids: Vec<usize>,
}

pub fn hamerly_kmeans(
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
    points: Vec<Vec<f64>>,
) -> HamerlyKmeansResult {
    if points.is_empty() {
        return HamerlyKmeansResult {
            centroids: Vec::new(),
            iterations: 0,
            point_centroids: Vec::new(),
        };
    }

    let mut centroids = get_centroids(&points, k);

    let InitializeResult {
        mut centroid_points_counts,
        mut centroid_points_sum,
        mut upper_bounds,
        mut lower_bounds,
        mut point_centroids,
    } = initialize(&centroids, &points);

    let mut centroid_closest_centroid_distance = vec![f64::MAX; centroids.len()];
    let mut centroid_distance_to_previous_position = vec![f64::MAX; centroids.len()];
    let mut iterations = 0;

    while iterations < max_iter {
        for (j, centroid) in centroids.iter().enumerate() {
            centroid_closest_centroid_distance[j] = get_min_centroid_skip_point_centroid(
                centroid,
                &centroids,
                j,
            )
            .1;
        }

        for (((point, point_centroid), lower_bound), upper_bound) in points
            .iter()
            .zip(point_centroids.iter_mut())
            .zip(lower_bounds.iter_mut())
            .zip(upper_bounds.iter_mut())
        {
            let m = f64::max(
                centroid_closest_centroid_distance[*point_centroid] / 2.0,
                *lower_bound,
            );
            if *upper_bound > m {
                *upper_bound = get_distance(point, &centroids[*point_centroid]);
                if *upper_bound > m {
                    let previous_point_centroid = *point_centroid;
                    point_all_centers(point, &centroids, upper_bound, lower_bound, point_centroid);
                    if previous_point_centroid != *point_centroid {
                        centroid_points_counts[previous_point_centroid] -= 1;
                        for (point_part, previous_point_centroid_sum_part) in point
                            .iter()
                            .zip(centroid_points_sum[previous_point_centroid].iter_mut())
                        {
                            *previous_point_centroid_sum_part -= point_part;
                        }
                        for (point_part, current_point_centroid_sum_part) in point
                            .iter()
                            .zip(centroid_points_sum[*point_centroid].iter_mut())
                        {
                            *current_point_centroid_sum_part += point_part;
                        }
                        centroid_points_counts[*point_centroid] += 1;
                    }
                }
            }
        }

        let total_squared_distance_moved = move_centers(
            &centroid_points_sum,
            &centroid_points_counts,
            &mut centroids,
            &mut centroid_distance_to_previous_position,
        );
        update_bounds(
            &centroid_distance_to_previous_position,
            &point_centroids,
            &mut upper_bounds,
            &mut lower_bounds,
        );

        if total_squared_distance_moved < convergence_threshold {
            break;
        }

        iterations += 1;
    }

    HamerlyKmeansResult {
        centroids,
        iterations,
        point_centroids,
    }
}

fn update_bounds(
    centroid_distance_to_previous_position: &[f64], // p(j) – distance that c(j) last moved
    point_centroids: &[usize], // a(i) index of the center to which x(i) is assigned
    upper_bounds: &mut [f64], // u(i) upper bound on the distance between x(i) and its assigned center c(a(i)),
    lower_bounds: &mut [f64], // l(i) lower bound on the distance between x(i) and its second closest center – that is, the closest center to x(i) that is not c(a(i))
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

    for ((upper_bound, lower_bound), point_centroid) in upper_bounds
        .iter_mut()
        .zip(lower_bounds.iter_mut())
        .zip(point_centroids.iter())
    {
        *upper_bound += centroid_distance_to_previous_position[*point_centroid];
        if r == *point_centroid {
            *lower_bound -= centroid_distance_to_previous_position[r_another];
        } else {
            *lower_bound -= centroid_distance_to_previous_position[r];
        }
    }
}

fn move_centers(
    centroid_points_sum: &[Vec<f64>],
    centroid_points_counts: &[usize],
    centroids: &mut [Vec<f64>],
    centroid_distance_to_previous_position: &mut [f64],
) -> f64 {
    let mut total_squared_distance_moved = 0.0;
    for (((centroid, points_sum), points_count), distance_to_previous_position) in centroids
        .iter_mut()
        .zip(centroid_points_sum.iter())
        .zip(centroid_points_counts.iter())
        .zip(centroid_distance_to_previous_position.iter_mut())
    {
        let mut squared_distance_moved = 0.0;
        for (centroid_part, points_sum_part) in centroid.iter_mut().zip(points_sum.iter()) {
            let previous_position = *centroid_part;
            *centroid_part = points_sum_part / *points_count as f64;
            squared_distance_moved += (previous_position - *centroid_part).powi(2);
        }
        *distance_to_previous_position = squared_distance_moved.sqrt();
        total_squared_distance_moved += squared_distance_moved;
    }
    total_squared_distance_moved
}

struct InitializeResult {
    centroid_points_counts: Vec<usize>, // q(j) – number of points assigned to cluster j
    centroid_points_sum: Vec<Vec<f64>>, // c`(j) vector sum of all points in cluster j
    upper_bounds: Vec<f64>, // u(i) upper bound on the distance between x(i) and its assigned center c(a(i)),
    lower_bounds: Vec<f64>, // l(i) lower bound on the distance between x(i) and its second closest center – that is, the closest center to x(i) that is not c(a(i))
    point_centroids: Vec<usize>, // a(i) index of the center to which x(i) is assigned
}
fn initialize(centroids: &[Vec<f64>], points: &[Vec<f64>]) -> InitializeResult {
    let mut centroid_points_counts: Vec<usize>;
    let mut centroid_points_sum: Vec<Vec<f64>>;
    let mut upper_bounds: Vec<f64>;
    let mut lower_bounds: Vec<f64>;
    let mut point_centroids: Vec<usize>;

    upper_bounds = vec![0.0; points.len()];
    lower_bounds = vec![0.0; points.len()];
    point_centroids = vec![0; points.len()];

    centroid_points_counts = vec![0; centroids.len()];
    centroid_points_sum = vec![vec![0.0; centroids[0].len()]; centroids.len()];
    for (((point, point_centroid), lower_bound), upper_bound) in points
        .iter()
        .zip(point_centroids.iter_mut())
        .zip(lower_bounds.iter_mut())
        .zip(upper_bounds.iter_mut())
    {
        point_all_centers(point, centroids, upper_bound, lower_bound, point_centroid);
        centroid_points_counts[*point_centroid] += 1;
        for (point_centroid_sum_part, point_part) in centroid_points_sum[*point_centroid]
            .iter_mut()
            .zip(point.iter())
        {
            *point_centroid_sum_part += point_part;
        }
    }

    InitializeResult {
        centroid_points_counts,
        centroid_points_sum,
        upper_bounds,
        lower_bounds,
        point_centroids,
    }
}

#[inline]
fn point_all_centers(
    point: &[f64],
    centroids: &[Vec<f64>],
    upper_bound: &mut f64,
    lower_bound: &mut f64,
    point_centroid: &mut usize,
) {
    (*point_centroid, *upper_bound) = get_min_centroid(point, centroids);
    (_, *lower_bound) = get_min_centroid_skip_point_centroid(
        point,
        centroids,
        *point_centroid,
    );
}

fn get_min_centroid(
    point: &[f64],
    centroids: &[Vec<f64>],
) -> (usize, f64) {
    let mut min_distance_squared = f64::MAX;
    let mut min_index = 0;

    for (j, centroid) in centroids.iter().enumerate() {
        let distance_squared = get_distance_squared(point, centroid);
        if distance_squared < min_distance_squared {
            min_distance_squared = distance_squared;
            min_index = j;
        }
    }

    (min_index, min_distance_squared.sqrt())
}

fn get_min_centroid_skip_point_centroid(
    point: &[f64],
    centroids: &[Vec<f64>],
    point_centroid: usize,
) -> (usize, f64) {
    let mut min_distance_squared = f64::MAX;
    let mut min_index = 0;

    for (j, centroid) in centroids.iter().enumerate() {
        if j == point_centroid {
            continue;
        }

        let distance_squared = get_distance_squared(point, centroid);
        if distance_squared < min_distance_squared {
            min_distance_squared = distance_squared;
            min_index = j;
        }
    }

    (min_index, min_distance_squared.sqrt())
}

fn get_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (*a - *b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn get_distance_squared(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (*a - *b).powi(2))
        .sum::<f64>()
}

#[cfg(target_arch = "wasm32")]
use js_sys::Math;
#[cfg(not(target_arch = "wasm32"))]
use rand::prelude::{SliceRandom, StdRng};
#[cfg(not(target_arch = "wasm32"))]
use rand::SeedableRng;

fn get_centroids(points: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut centroids = Vec::with_capacity(k);

    #[cfg(target_arch = "wasm32")]
    {
        let mut chosen_indices = Vec::with_capacity(k);
        while centroids.len() < k {
            let random_index = (Math::random() * (points.len() as f64)) as usize;
            if !chosen_indices.contains(&random_index) {
                centroids.push(points[random_index].clone());
                chosen_indices.push(random_index);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut rng = StdRng::seed_from_u64(0);
        centroids = points.choose_multiple(&mut rng, k).cloned().collect();
    }

    centroids
}

