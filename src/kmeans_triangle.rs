use rand::prelude::{SliceRandom, StdRng};
use rand::SeedableRng;

pub fn hamerly_kmeans(
    k: usize,
    max_iter: usize,
    convergence_threshold: f64,
    points: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    if points.is_empty() {
        return vec![];
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
    let mut iteration = 0;

    while iteration < max_iter {
        for j in 0..centroids.len() {
            centroid_closest_centroid_distance[j] = get_min_centroid(
                &centroids[j],
                centroids
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != j)
                    .map(|(_, centroid)| centroid),
            )
            .1;
        }

        for i in 0..points.len() {
            let m = f64::max(
                centroid_closest_centroid_distance[point_centroids[i]] / 2.0,
                lower_bounds[i],
            );
            if upper_bounds[i] > m {
                upper_bounds[i] = get_distance(&points[i], &centroids[point_centroids[i]]);
                if upper_bounds[i] > m {
                    let previous_point_centroid = point_centroids[i];
                    point_all_centers(
                        i,
                        &points,
                        &centroids,
                        &mut upper_bounds,
                        &mut lower_bounds,
                        &mut point_centroids,
                    );
                    if previous_point_centroid != point_centroids[i] {
                        centroid_points_counts[previous_point_centroid] -= 1;
                        for j in 0..points[i].len() {
                            centroid_points_sum[previous_point_centroid][j] -= points[i][j];
                            centroid_points_sum[point_centroids[i]][j] += points[i][j];
                        }
                        centroid_points_counts[point_centroids[i]] += 1;
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

        iteration += 1;
    }

    centroids
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

    for i in 0..upper_bounds.len() {
        upper_bounds[i] += centroid_distance_to_previous_position[point_centroids[i]];
        if r == point_centroids[i] {
            lower_bounds[i] -= centroid_distance_to_previous_position[r_another];
        } else {
            lower_bounds[i] -= centroid_distance_to_previous_position[r];
        }
    }
}

fn move_centers(
    centroid_points_sum: &[Vec<f64>],
    centroid_points_counts: &[usize],
    centroids: &mut Vec<Vec<f64>>,
    centroid_distance_to_previous_position: &mut [f64],
) -> f64 {
    let mut total_squared_distance_moved = 0.0;
    for j in 0..centroids.len() {
        let mut squared_distance_moved = 0.0;
        for i in 0..centroids[j].len() {
            let previous_position = centroids[j][i];
            centroids[j][i] = centroid_points_sum[j][i] / centroid_points_counts[j] as f64;
            squared_distance_moved += (previous_position - centroids[j][i]).powi(2);
        }
        centroid_distance_to_previous_position[j] = squared_distance_moved.sqrt();
        total_squared_distance_moved += squared_distance_moved;
    }
    total_squared_distance_moved
}

fn get_centroids(points: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(0);
    points.choose_multiple(&mut rng, k).cloned().collect()
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
    for i in 0..points.len() {
        point_all_centers(
            i,
            points,
            centroids,
            &mut upper_bounds,
            &mut lower_bounds,
            &mut point_centroids,
        );
        centroid_points_counts[point_centroids[i]] += 1;
        for j in 0..points[i].len() {
            centroid_points_sum[point_centroids[i]][j] += points[i][j];
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

fn point_all_centers(
    i: usize,
    points: &[Vec<f64>],
    centroids: &[Vec<f64>],
    upper_bounds: &mut [f64],
    lower_bounds: &mut [f64],
    point_centroids: &mut [usize],
) {
    let (point_centroid, upper_bound) = get_min_centroid(&points[i], centroids.iter());
    let (_, lower_bound) = get_min_centroid(
        &points[i],
        centroids
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != point_centroid)
            .map(|(_, centroid)| centroid),
    );

    upper_bounds[i] = upper_bound;
    lower_bounds[i] = lower_bound;
    point_centroids[i] = point_centroid;
}

fn get_min_centroid<'a>(
    point: &[f64],
    centroids: impl Iterator<Item = &'a Vec<f64>>,
) -> (usize, f64) {
    let mut min_distance = f64::MAX;
    let mut min_index = 0;

    for (j, centroid) in centroids.enumerate() {
        let distance = get_distance(point, centroid);
        if distance < min_distance {
            min_distance = distance;
            min_index = j;
        }
    }

    (min_index, min_distance)
}

fn get_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (*a - *b).powi(2))
        .sum::<f64>()
        .sqrt()
}
