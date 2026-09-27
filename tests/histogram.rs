//! Is the reduction to distinct colours actually exact, and does the bail-out
//! behave?

use kmeans_wasm::kmeans_triangle::{Optional, hamerly_kmeans_dispatched_weighted};
use kmeans_wasm::packed_histogram::collapse;

/// Sum of squared distances from every original pixel to its nearest centroid.
fn inertia(pixels: &[f64], centroids: &[f64], dimensions: usize) -> f64 {
    let mut total = 0.0;
    for pixel in pixels.chunks_exact(dimensions) {
        let mut best = f64::INFINITY;
        for centroid in centroids.chunks_exact(dimensions) {
            let mut sum = 0.0;
            for part in 0..dimensions {
                let difference = pixel[part] - centroid[part];
                sum += difference * difference;
            }
            if sum < best {
                best = sum;
            }
        }
        total += best;
    }
    total
}

fn as_f64(slice: &[u8]) -> Vec<f64> {
    slice.iter().map(|value| f64::from(*value)).collect()
}

/// The histogram must account for every pixel exactly once, and the distinct
/// values it reports must be the ones present.
#[test]
fn collapse_counts_every_pixel_once() {
    let pixels: Vec<[u8; 3]> = vec![
        [10, 20, 30],
        [10, 20, 30],
        [10, 20, 30],
        [10, 20, 30],
        [10, 20, 30],
        [200, 100, 50],
        [200, 100, 50],
        [0, 0, 0],
        [255, 255, 255],
    ];
    let flat: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();

    let collapsed = collapse(&flat, 3).expect("this input collapses");

    assert_eq!(collapsed.distinct, 4, "four distinct colours");
    assert_eq!(
        collapsed.weights.iter().sum::<f64>(),
        pixels.len() as f64,
        "weights must sum to the pixel count"
    );

    let mut found: Vec<(Vec<u8>, f64)> = collapsed
        .points
        .as_chunks::<3>()
        .0
        .iter()
        .zip(collapsed.weights.iter())
        .map(|(colour, weight)| {
            (
                colour.iter().map(|value| *value as u8).collect::<Vec<u8>>(),
                *weight,
            )
        })
        .collect();
    found.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(
        found,
        vec![
            (vec![0, 0, 0], 1.0),
            (vec![10, 20, 30], 5.0),
            (vec![200, 100, 50], 2.0),
            (vec![255, 255, 255], 1.0),
        ]
    );
}

/// The reduction is only sound if a weighted centroid equals the mean of the
/// pixels it stands for. With two well separated colours and k=2, each cluster
/// must land exactly on one colour's mean, weighted or not.
#[test]
fn weighted_centroids_equal_the_mean_of_the_pixels_they_stand_for() {
    let mut flat = Vec::new();
    for _ in 0..700 {
        flat.extend_from_slice(&[20, 40, 60]);
    }
    for _ in 0..300 {
        flat.extend_from_slice(&[220, 180, 140]);
    }

    let collapsed = collapse(&flat, 3).expect("two colours collapse");
    assert_eq!(collapsed.distinct, 2);

    let result = hamerly_kmeans_dispatched_weighted(
        2,
        50,
        0.0,
        &collapsed.points,
        3,
        Optional {
            weights: Some(&collapsed.weights),
            ..Optional::default()
        },
    );

    let mut centroids = result
        .centroids
        .as_chunks::<3>()
        .0
        .iter()
        .map(|centroid| centroid.to_vec())
        .collect::<Vec<_>>();
    centroids.sort_by(|a, b| a[0].total_cmp(&b[0]));

    assert_eq!(centroids.len(), 2);
    assert_eq!(centroids[0], vec![20.0, 40.0, 60.0]);
    assert_eq!(centroids[1], vec![220.0, 180.0, 140.0]);
}

/// The claim being checked: clustering distinct colours with weights gives the
/// same answer, bit for bit, as clustering every pixel.
///
/// Both runs start from the same `k` colours, which is the only way to separate a
/// difference in the algorithm from two different local optima.
///
/// The result is exact rather than merely close, and the reason is worth writing
/// down. The centroid update sums `w * v` once per distinct colour in the reduced
/// form and `1 * v` once per pixel in the full form, so the two accumulate in
/// different orders and in general that would change the last bits. Here it
/// cannot: every value is a `u8`, so every partial sum is a small integer, well
/// under the 2^53 where f64 stops representing integers exactly, and integer
/// addition is associative. The two forms compute the same integer sum, divide by
/// the same integer count, and land on the same `f64`.
///
/// This is specific to the packed entry points. The general `kmeans` API takes
/// `f64` values, where collapsing would only be mathematically exact, not
/// bit-exact, and it has nothing to collapse anyway since `f64` values are
/// effectively all distinct.
#[test]
fn the_reduction_is_exact_from_the_same_starting_colours() {
    // A gradient, not flat regions. Flat regions let a cluster settle on a
    // single colour, where the mean is an exact integer for a boring reason; a
    // gradient forces means that are awkward rationals.
    let mut data = Vec::with_capacity(320 * 240 * 3);
    for y in 0..240usize {
        for x in 0..320usize {
            for channel in 0..3usize {
                let value =
                    (x as f64 / 320.0) * 200.0 + (y as f64 / 240.0) * 50.0 + channel as f64 * 21.0;
                data.push(value as u8);
            }
        }
    }
    let collapsed = collapse(&data, 3).expect("a gradient still collapses");
    let wide = as_f64(&data);
    println!(
        "EXACT {} pixels collapsed to {} distinct, a {:.0}x smaller point set",
        data.len() / 3,
        collapsed.distinct,
        (data.len() / 3) as f64 / collapsed.distinct as f64
    );

    for k in [2usize, 4, 8, 16, 32] {
        // Start from k distinct colours taken out of the collapsed set.
        let step = (collapsed.distinct / k).max(1);
        let mut seeds = Vec::with_capacity(k * 3);
        for pick in (0..collapsed.distinct).step_by(step).take(k) {
            seeds.extend_from_slice(&collapsed.points[pick * 3..pick * 3 + 3]);
        }
        assert_eq!(seeds.len(), k * 3);

        let full = hamerly_kmeans_dispatched_weighted(
            k,
            100,
            0.0,
            &wide,
            3,
            Optional {
                initial_centroids: Some(&seeds),
                ..Optional::default()
            },
        );
        let reduced = hamerly_kmeans_dispatched_weighted(
            k,
            100,
            0.0,
            &collapsed.points,
            3,
            Optional {
                weights: Some(&collapsed.weights),
                initial_centroids: Some(&seeds),
            },
        );

        assert_eq!(
            full.centroids, reduced.centroids,
            "k={k}: the reduction must return the same centroids bit for bit"
        );
        assert_eq!(
            full.iterations, reduced.iterations,
            "k={k}: both forms should also take the same number of rounds"
        );
        assert_eq!(
            inertia(&wide, &full.centroids, 3),
            inertia(&wide, &reduced.centroids, 3)
        );
    }
}

/// Input with almost no duplicates must decline the reduction rather than pay for
/// a table that saves nothing, and must not silently truncate.
#[test]
fn indecomposable_input_declines_the_reduction() {
    let mut state = 0x1234_5678u64;
    let mut flat = Vec::with_capacity(1024 * 1024 * 3);
    for _ in 0..1024 * 1024 * 3 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        flat.push(((state >> 33) % 256) as u8);
    }

    assert!(
        collapse(&flat, 3).is_none(),
        "a million distinct colours should exceed the table and bail out"
    );
}

/// Input that an evenly spaced sample cannot see.
///
/// This is round-robin repetition with a period far longer than a sample stride.
/// A colour repeats every `distinct` pixels, and with 4096 evenly spaced samples
/// 117 pixels apart, two samples can only land on the same colour if they are
/// 40,000 samples apart, which never happens. So a sampling pre-check measured
/// zero duplicates on input that is genuinely 90% duplicate, and the reduction
/// declined it.
///
/// The reduction has to decide from the distinct count alone, so long-period
/// repetition that fits the table must be collapsed. The table is 2^18 slots and
/// is given up at three quarters full, so 196,608 distinct values.
#[test]
fn long_period_repetition_is_not_missed() {
    let pixels = 480_000usize;
    let load_limit = 196_608;

    for distinct in [150_000usize, 90_000, 48_000] {
        assert!(distinct <= load_limit, "this case must fit the table");
        let data = round_robin(pixels, distinct);

        let collapsed = collapse(&data, 3).unwrap_or_else(|| {
            panic!(
                "{:.0}% duplicate input with a period of {distinct} was declined",
                100.0 * (1.0 - distinct as f64 / pixels as f64)
            )
        });
        assert_eq!(collapsed.distinct, distinct, "every colour accounted for");
        assert_eq!(
            collapsed.weights.iter().sum::<f64>(),
            pixels as f64,
            "weights must sum to the pixel count"
        );
    }

    // Above the table's capacity declining is correct, not a miss: there are
    // simply more distinct values than there is room to count.
    assert!(collapse(&round_robin(pixels, 240_000), 3).is_none());
}

/// `pixels` pixels using exactly `distinct` colours, cycling, so repeats are as
/// far apart as the input allows.
///
/// The palette is a bijection on the index rather than a random triple, because
/// random triples collide: 90,000 of them would come to 89,754 distinct by
/// birthday, which makes an exact assertion impossible.
fn round_robin(pixels: usize, distinct: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(pixels * 3);
    for index in 0..pixels {
        let colour = index % distinct;
        data.push((colour / 65536) as u8);
        data.push((colour / 256 % 256) as u8);
        data.push((colour % 256) as u8);
    }
    data
}

/// The same claim, over many random starts rather than one.
///
/// Bit-identity from a single start could be luck: that start might happen to
/// give every cluster a single colour, where any summation order agrees. Twenty
/// random starts on a gradient, where the means are awkward rationals, is a real
/// test of the reduction.
#[test]
fn the_reduction_is_exact_from_many_random_starts() {
    let mut data = Vec::with_capacity(240 * 180 * 3);
    for y in 0..180usize {
        for x in 0..240usize {
            for channel in 0..3usize {
                let value =
                    (x as f64 / 240.0) * 210.0 + (y as f64 / 180.0) * 45.0 + channel as f64 * 13.0;
                data.push(value as u8);
            }
        }
    }
    let collapsed = collapse(&data, 3).expect("a gradient still collapses");
    let wide = as_f64(&data);

    let mut state = 0xfeed_face_cafe_beefu64;
    let next = move |state: &mut u64| {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % collapsed.distinct
    };

    let mut checked = 0;
    for k in [3usize, 6, 12, 24] {
        for _ in 0..5 {
            // k distinct colours out of the collapsed set.
            let mut seeds: Vec<f64> = Vec::with_capacity(k * 3);
            for _ in 0..k {
                let pick = next(&mut state);
                seeds.extend_from_slice(&collapsed.points[pick * 3..pick * 3 + 3]);
            }

            let full = hamerly_kmeans_dispatched_weighted(
                k,
                100,
                0.0,
                &wide,
                3,
                Optional {
                    initial_centroids: Some(&seeds),
                    ..Optional::default()
                },
            );
            let reduced = hamerly_kmeans_dispatched_weighted(
                k,
                100,
                0.0,
                &collapsed.points,
                3,
                Optional {
                    weights: Some(&collapsed.weights),
                    initial_centroids: Some(&seeds),
                },
            );

            assert_eq!(
                full.centroids, reduced.centroids,
                "k={k}: centroids differ from a random start"
            );
            assert_eq!(full.iterations, reduced.iterations, "k={k}: rounds differ");
            checked += 1;
        }
    }
    println!("MANY checked {checked} random starts, all bit-identical");
}
