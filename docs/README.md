# RGB and RGBA colorization playground

This directory contains a small GitHub Pages comparison for `kmeans-wasm`.

The page loads one public-domain Blue Marble image and renders three 16-color results: the packed
`kmeans_rgb` WebAssembly method, the four-component `kmeans_rgba` WebAssembly method, and the
MIT-licensed `skmeans` browser bundle. It also reports a median speed comparison between the three
implementations. It deliberately does not include a larger application UI or image-upload controls.

`kmeans_rgba` clusters `ImageData.data` directly, so the page passes the canvas pixels straight from
`getImageData` to WebAssembly without repacking them. The demo image is fully opaque, so its palette
matches the `kmeans_rgb` one; the extra time it reports is the cost of clustering the alpha channel as
a fourth component.

## Local preview

Build the browser WebAssembly package before serving the page:

```sh
npm ci
npm run pages:build
npx serve docs
```

The generated `docs/wasm/` directory is intentionally ignored. The page loads the release
WebAssembly build from that directory and uses `assets/blue-marble.jpg` as its single example.

## Benchmark

The benchmark tables in the repository README are reference runs. To reproduce them locally with the
current Node.js/npm toolchain, run:

```sh
npm ci
npm run bench:rgb
npm run bench:kmeans
```

The first command measures `kmeans_rgb` against `skmeans` on deterministic RGB point sets. The second
measures the general `kmeans` API across point counts, dimensions, and cluster counts, and also
reports `kmeans_rgb` on the three-dimensional cases so the cost of the generic array-based call is
visible. Both print a terminal table and a Markdown table. The native Criterion benchmarks cover both
packed color paths:

```sh
cargo bench --bench kmeans_rgb
cargo bench --bench kmeans_rgba
```

## Deployment

The `GitHub Pages` workflow builds `docs/wasm` and deploys the `docs/` directory on pushes to
`main`. GitHub Pages must be configured to use the `GitHub Actions` source in the repository
settings.
