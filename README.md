# kmeans-wasm

A fast k-means clustering implementation written in Rust and compiled to WebAssembly. It supports color quantization and general vector-space data, with JavaScript and TypeScript bindings.

Version 3 uses WebAssembly SIMD (`simd128`) for performance. Use a runtime with SIMD support, such as Chrome 91+, Firefox 89+, or Safari 16.4+.

## Features

- Hamerly k-means algorithm
- RGB color quantization
- Arbitrary numeric vector spaces
- JavaScript and TypeScript bindings
- ES module package with a documented `exports` entry point

## Installation

```sh
npm install kmeans-wasm
```

The published package targets JavaScript bundlers and exposes an ES module.

## Usage

### General vector spaces

```js
import { kmeans } from "kmeans-wasm";

const data = [
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
];

const result = kmeans(data, 3, 1_000, 0.001);

console.log(result.centroids);
console.log(result.idxs);
```

`kmeans(data, k, maxIter, convergenceThreshold?)` returns an object containing:

- `k`: the requested cluster count
- `it`: the number of completed iterations
- `centroids`: the calculated centroids
- `idxs`: a `Uint32Array` mapping each input point to a centroid
- `test`: a helper that assigns a new point to the nearest centroid

### RGB color quantization

```js
import { kmeans_rgb } from "kmeans-wasm";

const rgb = new Uint8Array([
  255, 0, 0,
  0, 255, 0,
  0, 0, 255,
]);

const quantizedColors = kmeans_rgb(rgb, 3, 1_000, 0.001);
```

`kmeans_rgb` returns a `Uint8Array` containing the RGB centroids.

## Development

Prerequisites:

- Rust 1.86 or newer
- Node.js 22.22.2+, 24.15.0+, or 26.0.0+
- npm 12.1.0
- Chrome or Chromium for browser tests

Install the toolchain and dependencies:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack --version 0.15.0 --locked
npm ci
```

Run the native, package, and TypeScript compatibility checks:

```sh
npm run check
```

Run the browser test suite separately:

```sh
npm run test:web
```

The release build is written to `pkg/`. Run `npm run package:check` to rebuild it and validate it with `publint` and `attw`.

## Browser playground

The [GitHub Pages comparison](https://vusolapohvistr.github.io/kmeans-wasm/) shows one public-domain Blue Marble image clustered with both `kmeans_rgb` and `skmeans`. It builds the browser WebAssembly package into the ignored `docs/wasm/` directory:

```sh
npm ci
npm run pages:build
npx serve docs
```

The page is intentionally a single-image comparison rather than a full application. See the [comparison source and deployment notes](https://github.com/vusolapohvistr/kmeans-wasm/tree/main/docs) for details.

## Benchmarks

Run the RGB comparison locally with the same release WebAssembly build used by the demo:

```sh
npm ci
npm run bench:rgb
```

The benchmark uses deterministic RGB point sets, reports median wall time, and prints a table suitable for updating the page. The Rust Criterion benchmark is also available with `cargo bench --bench kmeans_rgb`.

The earlier 1,000-pixel example was too small to be representative: startup, input conversion, and measurement noise dominated the result. The native benchmark keeps the 3.1.0 benchmark shape and uses ten times as many points (10,000 random RGB pixels, 3 centroids, 1,000 maximum iterations). The JavaScript benchmark rebuilds the WASM artifact before every benchmark run. The single-image comparison page reports the browser clustering time for both implementations.

### Reference RGB results

These are median wall times from a local Node.js 24.15.0 run using the release WebAssembly build, npm 12.1.0, two warm-ups, and eight measured runs. The browser playground shows the same table and is available at [vusolapohvistr.github.io/kmeans-wasm](https://vusolapohvistr.github.io/kmeans-wasm/).

| Test | Pixels | Colors | `kmeans_rgb` | `skmeans` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB random pixels | 1,000 | 2 | 0.20 ms | 1.07 ms | 5.3× |
| RGB random pixels | 10,000 | 4 | 1.21 ms | 4.43 ms | 3.7× |
| RGB random pixels | 10,000 | 16 | 3.54 ms | 8.94 ms | 2.5× |
| RGB random pixels | 100,000 | 2 | 4.67 ms | 20.33 ms | 4.4× |
| RGB random pixels | 100,000 | 8 | 19.24 ms | 51.99 ms | 2.7× |
| RGB random pixels | 100,000 | 32 | 81.57 ms | 138.80 ms | 1.7× |

`kmeans_rgb` is measured directly; `skmeans` receives the equivalent three-dimensional points. Results vary by CPU, browser, initialization, and convergence behavior. See the [reproducible benchmark harness](https://github.com/vusolapohvistr/kmeans-wasm/blob/main/js_bench/src/rgb.ts) for details. The package finalizer copies this README into `pkg/`, so the table is also visible on the npm package page.

## Releases

Releases use [`release-it`](https://github.com/release-it/release-it) locally. It reads and writes the version in `Cargo.toml`, refreshes `Cargo.lock`, runs the checks, commits the release, creates a `vX.Y.Z` tag, pushes it, and creates the GitHub Release.

After the GitHub Release is created, the `release:stage` hook runs `npm stage publish` against the generated `pkg/` directory. npm staging does not require a 2FA prompt; a maintainer must inspect the tarball and approve it with 2FA. This is compatible with npm's planned January 2027 removal of direct publishing through bypass-2FA GATs. There is no GitHub Actions npm publisher and no npm token in the repository.

Before releasing, authenticate locally with npm's web login:

```sh
npm login --auth-type=web --registry=https://registry.npmjs.org
npm whoami
```

Do not commit `.npmrc` or tokens. If a token is required, use an npm granular access token with **Read and write (stage only)** permissions for `kmeans-wasm`; never use a bypass-2FA token for direct publishing.

To release:

1. Merge the release changes to `main` and make sure CI is green.
2. Pull `main`, install dependencies, and preview the release:

   ```sh
   git switch main
   git pull --ff-only
   npm ci
   npm run release:dry
   ```

3. Start the release and choose the semantic version:

   ```sh
   npm run release
   ```

   Without `GITHUB_TOKEN`, `release-it` opens a prefilled GitHub Release page for manual confirmation; set a repository-scoped `GITHUB_TOKEN` when automated GitHub Release creation is preferred. After the release is created, the local hook stages the npm tarball.

4. Inspect and approve the staged npm package with 2FA:

   ```sh
   npm stage list kmeans-wasm
   npm stage download <stage-id>
   npm stage approve <stage-id>
   ```

Stable versions use the `latest` npm tag and prereleases use `next`. Reject a staged release with `npm stage reject <stage-id>` if manual inspection finds a problem.

## Comparison with skmeans

You can compare both libraries at <https://ycatbink0t.github.io/kmeans-web-comparison/>.

## Contributing

Pull requests and issues are welcome. Add tests for new features and bug fixes.

## License

[MIT](./LICENSE_MIT)
