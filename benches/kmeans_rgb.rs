use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{RngExt, rng};

use kmeans_wasm::*;

const MAX_ITERATIONS: usize = 100;
const CONVERGENCE_THRESHOLD: f64 = 0.1;

fn random_pixels(count: usize, rng: &mut impl rand::Rng) -> Vec<u8> {
    (0..count)
        .flat_map(|_| [rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()])
        .collect()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("kmeans_rgb");

    // Keep setup out of the measured function. The previous benchmark generated a
    // new random image inside every iteration, which made the result sensitive to
    // RNG and allocator noise rather than the clustering work.
    group
        .sample_size(30)
        .measurement_time(Duration::from_secs(5))
        .significance_level(0.1);

    for (pixel_count, color_count) in [(10_000, 4_usize), (100_000, 16), (409_600, 32)] {
        let pixels = random_pixels(pixel_count, &mut rng);
        let id = BenchmarkId::new("random_rgb", format!("{pixel_count}px_k{color_count}"));

        group.bench_with_input(id, &pixels, |b, pixels| {
            b.iter_batched(
                || pixels.clone(),
                |pixels| {
                    black_box(kmeans_rgb(
                        pixels,
                        color_count,
                        MAX_ITERATIONS,
                        Some(CONVERGENCE_THRESHOLD),
                    ))
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
