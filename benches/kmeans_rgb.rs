use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kmeans_wasm::*;
use rand::{thread_rng, Rng};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();
    let mut get_pixels = || {
        (0..1_000)
            .flat_map(|_| [rng.gen::<u8>(), rng.gen::<u8>(), rng.gen::<u8>()])
            .collect::<Vec<u8>>()
    };

    let mut group = c.benchmark_group("kmeans_rgb");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.significance_level(0.1).sample_size(1000);
    group.bench_function("10^3 random pixels, 3 centroids, kmeans_rgb", |b| {
        b.iter(|| kmeans_rgb(3, 1000, black_box(get_pixels())))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
