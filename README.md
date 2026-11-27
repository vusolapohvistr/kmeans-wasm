# kmeans-wasm

Fast, deterministic **k-means** (Hamerly) and stochastic **e-means** clustering in Rust compiled to WebAssembly (`simd128`), with zero-config ESM support for Node.js, Vite, Webpack, Next.js, and browsers.

[![npm version](https://img.shields.io/npm/v/kmeans-wasm.svg)](https://www.npmjs.com/package/kmeans-wasm)
[![CI](https://github.com/vusolapohvistr/kmeans-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/vusolapohvistr/kmeans-wasm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE_MIT)

- **Try it live:** [Interactive browser playground](https://vusolapohvistr.github.io/kmeans-wasm/) (upload any image, adjust palette size, toggle Floyd–Steinberg dithering, copy hex swatches)
- **Zero-config runtime support:** works out of the box in Node.js (ESM & `require`), Vite, Webpack, Next.js, and browsers—plus `./inline` (zero fetch/bundler config) and `./web` (explicit `init`)
- **Zero `eval` / CSP-safe:** works under strict Content Security Policies (Chrome Manifest V3, Electron, hardened SSR)
- **TypeScript-first:** full `.d.ts` declarations accepting both `Uint8Array` and `Uint8ClampedArray` (`ImageData.data`) directly

## Quick start

```js
import { kmeans_rgba, apply_palette } from "kmeans-wasm";

const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const palette = kmeans_rgba(imageData.data, 16, 30, 0.1, 42); // deterministic seed 42
const quantized = apply_palette(imageData.data, palette, 4);
```

## Installation

```sh
npm install kmeans-wasm
```

### Environments & bundlers

| Import path | Initialization | Best for |
| --- | --- | --- |
| `"kmeans-wasm"` | Automatic (Node `fs` or browser `fetch` + top-level `await`) | Node.js 20+, Vite, Webpack 5, Next.js, modern ESM browsers |
| `"kmeans-wasm/inline"` | Call `await init()` once (WASM embedded as base64, no `.wasm` asset fetch) | Environments without `.wasm` asset bundling or top-level `await` |
| `"kmeans-wasm/web"` | Call `await init(urlOrBytes)` explicitly | Custom WASM hosting, CDNs, or manual caching |

```js
// Zero-asset / zero-bundler-plugin usage (e.g. legacy bundlers or inline workers):
import init, { kmeans_rgb } from "kmeans-wasm/inline";
await init();
```

## Requirements

Uses WebAssembly SIMD (`simd128`), bulk memory, reference types, sign extension, non-trapping float-to-int conversion, and multi-value: Chrome 91+, Firefox 89+, Safari 16.4+, or Node.js 20+.

## Features

- **Exact Hamerly k-means** (`kmeans*`), bit-for-bit stable for a given seed
- **e-means mini-batch EMA** (`emeans*`), reaching comparable palettes in ~5 passes (~8.3× faster on photographs)
- **Distinct-color reduction** for packed RGB/RGBA buffers (Celebi weighted sort-means), speeding up real images by 5×–50× with bit-identical results
- **Palette mapping & dithering** (`apply_palette`, `apply_palette_indices`, `apply_palette_dithered` with Floyd–Steinberg error diffusion)
- **Flat TypedArray vector clustering** (`kmeans_f64`, `kmeans_f32`, `emeans_f64`, `emeans_f32`) avoiding JS nested-array marshaling overhead
- **Explicit deterministic `seed`** on every clustering entry point (no global `Math.random` patching needed)

## API overview

| Export | Input | Returns | Use it for |
| --- | --- | --- | --- |
| `kmeans_rgb` | `Uint8Array \| Uint8ClampedArray` (3/px) | `Uint8Array` | RGB image & palette work |
| `kmeans_rgba` | `Uint8Array \| Uint8ClampedArray` (4/px) | `Uint8Array` | `ImageData.data` directly (clusters alpha too) |
| `kmeans_f64` / `kmeans_f32` | `Float64Array` / `Float32Array` + `dimensions` | `{ k, it, dimensions, centroids, idxs }` | Fast vector/embedding clustering without nested arrays |
| `kmeans` | `Array<Array<number>>` | `{ k, it, centroids, idxs, test }` | Convenience API for nested JS arrays |
| `emeans_rgb` / `emeans_rgba` | `Uint8Array \| Uint8ClampedArray` | `Uint8Array` | Comparable palette in a fixed pass budget |
| `emeans_f64` / `emeans_f32` | `Float64Array` / `Float32Array` + `dimensions` | `{ k, it, dimensions, centroids, idxs }` | Fast stochastic clustering over flat typed vectors |
| `emeans` | `Array<Array<number>>` | `{ k, it, centroids, idxs, test }` | Stochastic clustering for nested JS arrays |
| `apply_palette` | `pixels`, `palette`, `components` (3 \| 4) | `Uint8Array` (RGBA) | Quantized pixels ready for `ImageData` |
| `apply_palette_indices` | `pixels`, `palette`, `components` (3 \| 4) | `Uint32Array` | Palette index per pixel (GIF/PNG-8/WebGL) |
| `apply_palette_dithered` | `pixels`, `width`, `height`, `palette`, `components` | `Uint8Array` (RGBA) | Floyd–Steinberg dithered quantization |

### Flat vector clustering (`Float64Array` / `Float32Array`)

```ts
import { kmeans_f32, emeans_f64 } from "kmeans-wasm";

// 4 vectors of 3 dimensions each, packed row-major
const vectors = new Float32Array([
  0, 0, 0,
  0.1, 0.2, 0.1,
  10, 10, 10,
  10.2, 9.9, 10.1,
]);

const result = kmeans_f32(vectors, 3, 2, 100, 1e-4, 42);
// result.centroids -> Float32Array(6), result.idxs -> Uint32Array(4)

const fast = emeans_f64(new Float64Array(vectors), 3, 2, 5, null, 42);
```

### Nested array vector spaces

```js
import { kmeans } from "kmeans-wasm";

const data = [
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
];

const result = kmeans(data, 3, 1_000, 0.001, 42);

console.log(result.centroids); // number[][]
console.log(result.idxs);      // Uint32Array

// Assign a new point (or pass a custom distance callback)
const cluster = result.test([2.5, 3.5]);
const byManhattan = result.test([2.5, 3.5], (a, b) =>
  a.reduce((sum, value, i) => sum + Math.abs(value - b[i]), 0),
);
```

### Quantizing & dithering an image in the browser

```js
import { kmeans_rgba, apply_palette, apply_palette_dithered } from "kmeans-wasm";

const context = canvas.getContext("2d", { willReadFrequently: true });
const image = context.getImageData(0, 0, canvas.width, canvas.height);

// Uint8ClampedArray (image.data) is accepted directly
const palette = kmeans_rgba(image.data, 16, 30, 0.1, 42);

// Exact nearest-palette mapping:
const flat = apply_palette(image.data, palette, 4);
// Or Floyd-Steinberg error-diffusion dithering:
const dithered = apply_palette_dithered(image.data, canvas.width, canvas.height, palette, 4);

const output = context.createImageData(canvas.width, canvas.height);
output.data.set(dithered);
context.putImageData(output, 0, 0);
```

`apply_palette` and `apply_palette_indices` resolve each distinct color once and map pixels with a table probe. Ties go to the lower palette index, matching the clustering core. Pass `3` with `kmeans_rgb` and `apply_palette` / `apply_palette_dithered` writes an opaque alpha (`255`).

### e-means: a pass budget instead of a convergence test

`emeans_rgb`, `emeans_rgba`, `emeans_f64`, `emeans_f32`, and `emeans` run the mini-batch EMA algorithm with a cosine-annealed step schedule:

```js
import { emeans_rgba } from "kmeans-wasm";

const palette = emeans_rgba(image.data, 16);              // 5 passes by default
const quicker = emeans_rgba(image.data, 16, 3);           // fewer passes
const pinned = emeans_rgba(image.data, 16, 5, 4096, 42);  // explicit batch size & seed
```

- `it` is the number of completed passes.
- `idxs` holds each point's assignment during the final pass; use `result.test(point)` (on `emeans`) if you need nearest-centroid queries against the finalized centroids.
- Passing `seed` makes both `kmeans*` and `emeans*` deterministic without mutating `Math.random`. When `seed` is omitted on WebAssembly, initial draws read `Math.random`.

### Arguments and errors

All entry points throw standard JavaScript `RangeError` or `TypeError` instances (`err instanceof Error`):

- `k` must be at least 2
- `dimensions` (for `*_f64`/`*_f32`) must be at least 1, and `data.length` a multiple of `dimensions`
- `max_iter` must be at least 1
- `epochs` must be at least 1
- `batch_size` must be at least 1
- `convergence_threshold` must be finite and non-negative
- All vector coordinates in `kmeans*` / `emeans*` must be finite numbers
- Packed RGB/RGBA buffers must have length divisible by 3 or 4
- `apply_palette*` requires `components` of 3 or 4, `pixels` length divisible by `components`, and non-empty `palette` divisible by `components` (`apply_palette_dithered` also requires `pixels.length === width * height * components`)

`kmeans_rgb` and `kmeans_rgba` default `convergence_threshold` to `0.1`, below the resolution of the `u8` result. Pass `0` to always run every iteration. `kmeans`, `kmeans_f64`, and `kmeans_f32` default `convergence_threshold` to `0`.

## Benchmarks

The reference tables come from the release WebAssembly build on Node.js 26.10.0,
measured as a median of eight calls after two warm-ups in one harness run. The
smallest rows are the noisiest because fixed overhead is a large share.

`skmeans` has no convergence threshold. The RGB and vector tables therefore give
both libraries no early exit, which is the conservative direction for
`kmeans_rgb`: in normal use it stops sooner. The real-image table gives
`skmeans` one measured run after one warm-up because it costs seconds per image;
every column is a full-image measurement, so the columns are comparable.

### Reference RGB results

100 maximum iterations.

| Test | Pixels | Colors | `kmeans_rgb` | `skmeans` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB random pixels | 1,000 | 2 | 0.50 ms | 0.55 ms | 1.1× |
| RGB random pixels | 10,000 | 4 | 6.38 ms | 27.80 ms | 4.4× |
| RGB random pixels | 10,000 | 16 | 9.64 ms | 57.65 ms | 6.0× |
| RGB random pixels | 100,000 | 2 | 54.62 ms | 40.00 ms | 0.7× |
| RGB random pixels | 100,000 | 8 | 95.44 ms | 225.54 ms | 2.4× |
| RGB random pixels | 100,000 | 32 | 140.08 ms | 1413.71 ms | 10.1× |

### Reference general vector-space results

Same setup, over points of varying dimension and cluster count.

| Points | Dimensions | Clusters | `kmeans` | `kmeans_rgb` | `skmeans` | Speed-up | Packed speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3 | 2 | 0.63 ms | 0.26 ms | 0.44 ms | 0.7× | 2.4× |
| 10,000 | 3 | 2 | 3.80 ms | 2.85 ms | 4.81 ms | 1.3× | 1.3× |
| 10,000 | 3 | 10 | 9.07 ms | 6.56 ms | 40.90 ms | 4.5× | 1.4× |
| 10,000 | 3 | 50 | 15.54 ms | 16.12 ms | 352.55 ms | 22.7× | 1.0× |
| 100,000 | 3 | 10 | 95.99 ms | 90.57 ms | 844.36 ms | 8.8× | 1.1× |
| 10,000 | 10 | 10 | 20.56 ms | — | 270.16 ms | 13.1× | — |
| 10,000 | 50 | 10 | 59.35 ms | — | 1244.95 ms | 21.0× | — |
| 10,000 | 50 | 50 | 169.39 ms | — | 3815.23 ms | 22.5× | — |

`Speed-up` is `skmeans` divided by `kmeans`. `Packed speed-up` is `kmeans`
divided by `kmeans_rgb` on the same three-dimensional points. Random points are
99.7% distinct, so the packed path saves the boundary copy rather than the
clustering work.

### Reference real-image results

Every column is a full-image measurement of the same pixels.

| Image | Pixels | Distinct | Colors | `kmeans_rgb` | `kmeans_rgba` | `skmeans` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blue-marble | 922,560 | 6.2% | 8 | 47.18 ms | 45.36 ms | 4028.71 ms | 85.4× |
| blue-marble | 922,560 | 6.2% | 16 | 78.71 ms | 76.88 ms | 10856.36 ms | 137.9× |
| blue-marble | 922,560 | 6.2% | 32 | 88.48 ms | 92.28 ms | 11549.45 ms | 130.5× |
| blue-marble | 922,560 | 6.2% | 64 | 96.01 ms | 106.54 ms | 24432.12 ms | 254.5× |
| city | 307,200 | 45.8% | 8 | 134.70 ms | 135.55 ms | 906.41 ms | 6.7× |
| city | 307,200 | 45.8% | 16 | 174.22 ms | 150.79 ms | 2211.73 ms | 12.7× |
| city | 307,200 | 45.8% | 32 | 212.35 ms | 200.40 ms | 4234.70 ms | 19.9× |
| city | 307,200 | 45.8% | 64 | 255.47 ms | 251.75 ms | 7923.90 ms | 31.0× |
| coast | 307,200 | 30.5% | 8 | 85.05 ms | 77.30 ms | 768.55 ms | 9.0× |
| coast | 307,200 | 30.5% | 16 | 118.89 ms | 110.49 ms | 2284.65 ms | 19.2× |
| coast | 307,200 | 30.5% | 32 | 150.16 ms | 152.19 ms | 4263.44 ms | 28.4× |
| coast | 307,200 | 30.5% | 64 | 178.74 ms | 187.40 ms | 8046.12 ms | 45.0× |
| harbour | 307,200 | 36.6% | 8 | 94.01 ms | 87.89 ms | 625.35 ms | 6.7× |
| harbour | 307,200 | 36.6% | 16 | 108.88 ms | 110.00 ms | 1993.29 ms | 18.3× |
| harbour | 307,200 | 36.6% | 32 | 156.69 ms | 164.25 ms | 4705.96 ms | 30.0× |
| harbour | 307,200 | 36.6% | 64 | 206.54 ms | 202.09 ms | 8298.47 ms | 40.2× |

### Reference e-means results

Lloyd is 100 iterations with no early convergence. e-means is five passes with
the default batch and step laws. Both sides are seeded from the same stream
before every call, so their initial centroids match. MSE is the weighted mean
squared nearest-palette distance over distinct colors, which is also the mean
over pixels.

| Image | Pixels | Colors | Lloyd MSE | e-means MSE | MSE ratio | `kmeans_rgb` | `emeans_rgb` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blue-marble | 922,560 | 8 | 272.84 | 221.00 | 0.810 | 47.46 ms | 9.95 ms | 4.8× |
| blue-marble | 922,560 | 16 | 103.46 | 95.65 | 0.925 | 76.41 ms | 11.74 ms | 6.5× |
| blue-marble | 922,560 | 32 | 56.08 | 50.25 | 0.896 | 87.55 ms | 13.23 ms | 6.6× |
| city | 307,200 | 8 | 958.12 | 958.06 | 1.000 | 135.21 ms | 17.24 ms | 7.8× |
| city | 307,200 | 16 | 497.16 | 485.54 | 0.977 | 164.04 ms | 19.87 ms | 8.3× |
| city | 307,200 | 32 | 270.24 | 267.15 | 0.989 | 236.95 ms | 23.52 ms | 10.1× |
| coast | 307,200 | 8 | 713.09 | 713.12 | 1.000 | 93.23 ms | 11.28 ms | 8.3× |
| coast | 307,200 | 16 | 348.70 | 348.46 | 0.999 | 119.15 ms | 14.87 ms | 8.0× |
| coast | 307,200 | 32 | 226.84 | 204.96 | 0.904 | 162.41 ms | 17.56 ms | 9.3× |
| harbour | 307,200 | 8 | 1062.40 | 1062.43 | 1.000 | 94.75 ms | 14.44 ms | 6.6× |
| harbour | 307,200 | 16 | 492.09 | 492.11 | 1.000 | 136.04 ms | 16.16 ms | 8.4× |
| harbour | 307,200 | 32 | 256.51 | 250.03 | 0.975 | 188.56 ms | 19.88 ms | 9.5× |

Median five-epoch speed-up is about `8.3x`; three epochs are faster, around `11.8x` median.

### Reference mapping results

The baseline is a per-pixel search over the palette, memoised by packed color.
Both columns are byte-identical on every row.

Median of eight measured calls after two warm-ups, one harness run.

| Image | Pixels | Distinct | Colors | `apply_palette` | JavaScript loop | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blue-marble | 922,560 | 6.2% | 8 | 8.63 ms | 23.10 ms | 2.68× |
| blue-marble | 922,560 | 6.2% | 16 | 6.96 ms | 18.91 ms | 2.72× |
| blue-marble | 922,560 | 6.2% | 32 | 9.08 ms | 22.62 ms | 2.49× |
| blue-marble | 922,560 | 6.2% | 64 | 9.59 ms | 26.29 ms | 2.74× |
| city | 307,200 | 45.8% | 8 | 6.91 ms | 15.32 ms | 2.22× |
| city | 307,200 | 45.8% | 16 | 6.53 ms | 18.46 ms | 2.83× |
| city | 307,200 | 45.8% | 32 | 7.84 ms | 25.94 ms | 3.31× |
| city | 307,200 | 45.8% | 64 | 8.49 ms | 32.50 ms | 3.83× |
| coast | 307,200 | 30.5% | 8 | 4.62 ms | 10.28 ms | 2.22× |
| coast | 307,200 | 30.5% | 16 | 5.10 ms | 12.08 ms | 2.37× |
| coast | 307,200 | 30.5% | 32 | 5.26 ms | 15.80 ms | 3.00× |
| coast | 307,200 | 30.5% | 64 | 6.10 ms | 21.55 ms | 3.53× |
| harbour | 307,200 | 36.6% | 8 | 6.18 ms | 9.77 ms | 1.58× |
| harbour | 307,200 | 36.6% | 16 | 5.71 ms | 11.88 ms | 2.08× |
| harbour | 307,200 | 36.6% | 32 | 7.61 ms | 20.30 ms | 2.67× |
| harbour | 307,200 | 36.6% | 64 | 7.36 ms | 25.64 ms | 3.48× |

Above roughly half distinct, a per-pixel search becomes competitive again. A
call on input that does not repeat at all costs about 0.83× because the table
build is wasted before the fallback search.

## Credits

The clustering core is Hamerly k-means, from Grant Hamerly and Charles Elkan,
*Efficient Algorithms for Clustering with Bounds*, *Data Mining and Knowledge
Discovery* 24(3), 581-606, 2010. It accelerates the iteration Stuart Lloyd
described in *Least Squares Quantization in PCM*, *IEEE Transactions on
Information Theory* 28(2), 129-137, 1982, [doi][lloyd].

The e-means entry points follow Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze
and Jégou, *e-means: outperforming k-means (in one-tenth the epochs)*,
[arXiv:2611.09501][emeans], and the reference implementation in Faiss
`contrib/e_means`.

Reducing packed input to its distinct colors before clustering follows M. Emre
Celebi, *Fast Color Quantization Using Weighted Sort-Means Clustering*, *JOSA A*
26(11), 2009, [doi][celebi2], and *Improving the Performance of K-Means for Color
Quantization*, [arXiv:1101.0395][celebi].

[lloyd]: https://doi.org/10.1109/TIT.1982.1056489
[emeans]: https://arxiv.org/abs/2611.09501
[celebi]: https://arxiv.org/abs/1101.0395
[celebi2]: https://doi.org/10.1364/JOSAA.26.002434

## Contributing

Pull requests and issues are welcome. Add tests for new features and bug fixes.
From version 3.2.0 onward, much of the implementation and test engineering was
built in collaboration with AI coding assistants.

## License

[MIT](./LICENSE_MIT)
