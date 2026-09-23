# RGB colorization playground

This directory contains a small GitHub Pages comparison for `kmeans-wasm`.

The page loads one public-domain Blue Marble image and renders two 16-color results: the dedicated
`kmeans_rgb` WebAssembly method and the MIT-licensed `skmeans` browser bundle. It deliberately does
not include a larger application UI or image-upload controls.

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

The benchmark table in the repository README is a reference run. To reproduce it locally with the
current Node.js/npm toolchain, run:

```sh
npm ci
npm run bench:rgb
```

The command measures `kmeans_rgb` against `skmeans` on deterministic RGB point sets and prints
both a terminal table and a Markdown table.

## Deployment

The `GitHub Pages` workflow builds `docs/wasm` and deploys the `docs/` directory on pushes to
`main`. GitHub Pages must be configured to use the `GitHub Actions` source in the repository
settings.
