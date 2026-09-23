# RGB colorization playground

This directory contains the GitHub Pages demo for `kmeans-wasm`.

## Local preview

Build the browser WebAssembly package before serving the page:

```sh
npm ci
npm run pages:build
npx serve docs
```

The generated `docs/wasm/` directory is intentionally ignored. The page loads the release
WebAssembly build from that directory and uses the public-domain Blue Marble image in
`assets/blue-marble.jpg` as its default example.

## Benchmark

The benchmark table in the page is a reference run. To reproduce it locally with the current
Node.js/npm toolchain, run:

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
