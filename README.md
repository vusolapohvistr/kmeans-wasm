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

The release build is written to `pkg/`. Run `npm run package:check` to rebuild it and validate it with `publint` and Are the Types Wrong?.

## Releases

Releases use [`release-it`](https://github.com/release-it/release-it) for the local release workflow and npm trusted publishing (OIDC) in GitHub Actions for publication. No npm access token is stored in the repository.

`release-it` reads and writes the version in `Cargo.toml`, refreshes `Cargo.lock`, runs the checks, commits the release, creates a `vX.Y.Z` tag, pushes it, and creates the GitHub Release. Its npm plugin is disabled intentionally: the release workflow is the only publisher and stages the generated package for 2FA approval.

A package maintainer must configure the trusted publisher once after the release workflow is merged:

```sh
npm trust github kmeans-wasm \
  --repo vusolapohvistr/kmeans-wasm \
  --file publish.yml \
  --env npm \
  --allow-stage-publish
```

Then select **Require two-factor authentication and disallow tokens** in the package's npm publishing settings.

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

   Without `GITHUB_TOKEN`, `release-it` opens a prefilled GitHub Release page for manual confirmation. Set a repository-scoped `GITHUB_TOKEN` when automated GitHub Release creation is preferred.

4. The `Stage npm release` workflow builds the package, verifies the tag/version match, and submits it to npm's staging queue. Inspect and approve it with 2FA:

   ```sh
   npm stage list kmeans-wasm
   npm stage download <stage-id>
   npm stage approve <stage-id>
   ```

Trusted publishing attaches npm provenance automatically. Reject a staged release with `npm stage reject <stage-id>` if manual inspection finds a problem.

## Comparison with skmeans

You can compare both libraries at <https://ycatbink0t.github.io/kmeans-web-comparison/>.

## Contributing

Pull requests and issues are welcome. Add tests for new features and bug fixes.

## License

[MIT](./LICENSE_MIT)
