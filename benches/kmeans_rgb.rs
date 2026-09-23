use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use kmeans_wasm::*;
use rand::{RngExt, rng};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    let mut get_pixels = || {
        (0..10_000)
            .flat_map(|_| [rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()])
            .collect::<Vec<u8>>()
    };

    let mut group = c.benchmark_group("kmeans_rgb");
    // Keep the 3.1.0 benchmark shape, with ten times as many input points.
    group.significance_level(0.1).sample_size(1000);
    group.bench_function("10^4 random pixels, 3 centroids, kmeans_rgb", |b| {
        b.iter(|| kmeans_rgb(black_box(get_pixels()), 3, 1000, Some(0.1)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
