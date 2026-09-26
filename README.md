# kmeans-wasm

A fast k-means clustering implementation written in Rust and compiled to WebAssembly. It supports color quantization and general vector-space data, with JavaScript and TypeScript bindings.

Version 3 uses WebAssembly SIMD (`simd128`) for performance. Use a runtime with SIMD support, such as Chrome 91+, Firefox 89+, or Safari 16.4+.

## Features

- Hamerly k-means algorithm
- RGB and RGBA color quantization
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

### RGBA color quantization

```js
import { kmeans_rgba } from "kmeans-wasm";

const rgba = new Uint8Array([
  255, 0, 0, 255,
  0, 255, 0, 255,
  0, 0, 255, 128,
]);

const quantizedColors = kmeans_rgba(rgba, 3, 1_000, 0.001);
```

`kmeans_rgba` mirrors `kmeans_rgb` for four-component vectors and returns a `Uint8Array` containing
the RGBA centroids. It is the drop-in choice for `ImageData.data` and other buffers that interleave
red, green, blue, and alpha, because no repacking is needed before clustering. The alpha channel is
clustered like any other component, so the function also works for data that mixes transparent and
opaque pixels.

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

The [GitHub Pages comparison](https://vusolapohvistr.github.io/kmeans-wasm/) shows one public-domain Blue Marble image clustered with `kmeans_rgb`, with `kmeans_rgba`, and with `skmeans`, and reports the measured time of each implementation. It builds the browser WebAssembly package into the ignored `docs/wasm/` directory:

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

Run the general vector-space comparison the same way:

```sh
npm ci
npm run bench:kmeans
```

Both benchmarks use deterministic point sets, report median wall time, and print a table suitable for updating this README. The Rust Criterion benchmarks are also available with `cargo bench --bench kmeans_rgb` and `cargo bench --bench kmeans_rgba`.

The earlier 1,000-pixel example was too small to be representative: startup, input conversion, and measurement noise dominated the result. The native benchmark now pre-generates its input and tests 10k, 100k, and 409,600 pixels; the JavaScript benchmark rebuilds the WASM artifact before every benchmark run. The single-image comparison page reports the browser clustering time for all three implementations and shows what the extra alpha component of `kmeans_rgba` costs next to `kmeans_rgb`.

### Reference RGB results

These are median wall times from a local Node.js 26.10.0 run on an AMD Ryzen 5 9600X, using the release WebAssembly build, npm 12.1.0, two warm-ups, and eight measured runs. Each cell is the median of five harness runs, because a single run on a shared machine is not reproducible to better than roughly 30%. The browser playground is available at [vusolapohvistr.github.io/kmeans-wasm](https://vusolapohvistr.github.io/kmeans-wasm/).

| Test | Pixels | Colors | `kmeans_rgb` | `skmeans` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB random pixels | 1,000 | 2 | 0.15 ms | 0.33 ms | 2.1× |
| RGB random pixels | 10,000 | 4 | 0.92 ms | 4.05 ms | 4.7× |
| RGB random pixels | 10,000 | 16 | 2.30 ms | 8.37 ms | 3.5× |
| RGB random pixels | 100,000 | 2 | 4.46 ms | 15.58 ms | 3.6× |
| RGB random pixels | 100,000 | 8 | 14.29 ms | 57.24 ms | 3.6× |
| RGB random pixels | 100,000 | 32 | 56.46 ms | 126.95 ms | 2.3× |

`kmeans_rgb` is measured directly; `skmeans` receives the equivalent three-dimensional points. Results vary by CPU, browser, initialization, and convergence behavior. See the [reproducible benchmark harness](https://github.com/vusolapohvistr/kmeans-wasm/blob/main/js_bench/src/rgb.ts) for details. The package finalizer copies this README into `pkg/`, so the table is also visible on the npm package page.

### Reference general vector-space results

These are median wall times from a local Node.js 26.10.0 run on an AMD Ryzen 5 9600X, using the release WebAssembly build and npm 12.1.0. Each cell is the median of five harness runs, and every harness run is itself a median of eight measured runs after two warm-ups, with 100 maximum iterations and a 0.1 convergence threshold. Every point is a deterministic pseudo-random vector of `u8`-range values.

| Points | Dimensions | Clusters | `kmeans` | `kmeans_rgb` | `skmeans` | Speed-up | Packed speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3 | 2 | 0.34 ms | 0.07 ms | 0.69 ms | 2.0× | 5.0× |
| 10,000 | 3 | 2 | 1.74 ms | 0.44 ms | 2.77 ms | 1.6× | 3.9× |
| 10,000 | 3 | 10 | 2.80 ms | 1.60 ms | 15.31 ms | 5.0× | 2.0× |
| 10,000 | 3 | 50 | 9.21 ms | 8.26 ms | 33.20 ms | 3.5× | 1.1× |
| 100,000 | 3 | 10 | 26.00 ms | 15.08 ms | 132.27 ms | 5.2× | 1.7× |
| 10,000 | 10 | 10 | 6.09 ms | — | 18.68 ms | 3.1× | — |
| 10,000 | 50 | 10 | 21.41 ms | — | 60.57 ms | 2.9× | — |
| 10,000 | 50 | 50 | 36.68 ms | — | 114.21 ms | 3.2× | — |

`Speed-up` is `skmeans` divided by `kmeans`. `Packed speed-up` is `kmeans` divided by `kmeans_rgb` on the same three-dimensional points, and is shown only where the packed three-component API applies. All three columns measure the same clustering problem; the general `kmeans` call has to copy each point across the JavaScript boundary, while `kmeans_rgb` receives one packed `Uint8Array`.

That interop cost is fixed per point, so it dominates the small cases and shrinks as the cluster count grows. It is also why the playground page only benchmarks the packed color paths: converting a 480px preview into one array of point arrays costs more than the clustering itself. Prefer `kmeans_rgb` or `kmeans_rgba` for three- and four-component data, and reserve `kmeans` for genuinely arbitrary dimensions. Results vary by CPU, runtime, initialization, and convergence behavior; the smallest rows are the noisiest because fixed overhead is a large share of them. See the [reproducible benchmark harness](https://github.com/vusolapohvistr/kmeans-wasm/blob/main/js_bench/src/vector.ts) for details.

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
