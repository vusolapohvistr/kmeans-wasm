# kmeans-wasm

A fast k-means clustering implementation written in Rust and compiled to WebAssembly. It supports color quantization and general vector-space data, with JavaScript and TypeScript bindings. It ships both the exact Hamerly k-means path and a stochastic e-means path that reaches a comparable palette in a handful of passes.

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

## A note on how this was written

From version **3.2.0** onward, most of the work on this library was done by AI
assistants. The algorithms and the hard ideas belong to the researchers credited
above; the bugs belong to the AI.

## Requirements

Version 3 uses WebAssembly SIMD (`simd128`). Use Chrome 91+, Firefox 89+, or
Safari 16.4+. Bulk memory, reference types, sign extension, non-trapping
float-to-int conversion, and multi-value are assumed.

## Features

- Hamerly k-means, bit-for-bit stable for a given random draw
- e-means mini-batch EMA clustering, with a fixed pass budget instead of a convergence test
- RGB and RGBA color quantization through either clustering family
- Distinct-color reduction for packed input, weighted by pixel count
- Palette mapping keyed on distinct colors rather than pixels
- Arbitrary numeric vector spaces
- SIMD inner loop, no `unsafe`, no runtime feature detection
- JavaScript and TypeScript bindings
- ES module package with a documented `exports` entry point

## Installation

```sh
npm install kmeans-wasm
```

The published package targets JavaScript bundlers and exposes an ES module.

## Usage

There are two clustering families. The `kmeans*` entry points run Hamerly
k-means, whose output is bit-for-bit stable for a given random draw. The
`emeans*` entry points run the mini-batch EMA method from the e-means paper:
stochastic, with a fixed pass budget instead of a convergence test. Both accept
the same input layouts, both return palettes in the same shape, and
`apply_palette` maps either one back onto an image.

| Export | Input | Use it for |
| --- | --- | --- |
| `kmeans_rgb` | `Uint8Array`, 3 values per point | image and palette work |
| `kmeans_rgba` | `Uint8Array`, 4 values per point | the same, when alpha matters |
| `kmeans` | `Array<Array<number>>` | any other number of dimensions |
| `emeans_rgb` | `Uint8Array`, 3 values per point | the same palette with fewer passes |
| `emeans_rgba` | `Uint8Array`, 4 values per point | the same, when alpha matters |
| `emeans` | `Array<Array<number>>` | e-means over any other dimension |
| `apply_palette` | `Uint8Array`, a palette | replacing each pixel with its nearest entry |

The packed entry points take a flat typed array and avoid copying points one at
a time across the JavaScript boundary. The general entry points are the only
option outside three or four values per point.

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
- `centroids`: the calculated centroids, an array of `k` arrays of `dimensions` numbers
- `idxs`: a `Uint32Array` mapping each input point to a centroid
- `test`: a helper that assigns a new point to the nearest centroid

All points must have the same number of values, otherwise the call throws.

`test` is a method on the result, so call it as `result.test(point)`:

```js
const result = kmeans(data, 3, 1_000, 0.001);

// Which cluster does a new 2D point belong to?
const cluster = result.test([2.5, 3.5]);

// Bring your own distance function if you are not working in plain Euclidean space.
const byManhattan = result.test([2.5, 3.5], (a, b) =>
  a.reduce((sum, value, i) => sum + Math.abs(value - b[i]), 0),
);
```

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

`kmeans_rgb` returns a `Uint8Array` of RGB centroids. The input length must be a
multiple of three.

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

`kmeans_rgba` mirrors `kmeans_rgb` with four components per point and returns
RGBA centroids. It accepts `ImageData.data` directly, and clusters alpha like any
other channel. The input length must be a multiple of four.

### Quantizing an image in the browser

```js
import { kmeans_rgba, apply_palette } from "kmeans-wasm";

const context = canvas.getContext("2d", { willReadFrequently: true });
const image = context.getImageData(0, 0, canvas.width, canvas.height);

const palette = kmeans_rgba(image.data, 16, 30, 0.1);

const output = context.createImageData(canvas.width, canvas.height);
output.data.set(apply_palette(image.data, palette, 4));
context.putImageData(output, 0, 0);
```

`apply_palette` resolves each distinct color once and then maps pixels with a
table probe. A tie goes to the lower palette index, matching the clustering core,
so a hand-written search with a strict `<` produces the same bytes. Pass `3` with
`kmeans_rgb` and it writes an opaque alpha.

Run a full working version, including a three-way speed comparison against
`kmeans_rgb` and `skmeans`, at
[vusolapohvistr.github.io/kmeans-wasm](https://vusolapohvistr.github.io/kmeans-wasm/).

### e-means: a pass budget instead of a convergence test

`emeans_rgb`, `emeans_rgba`, and `emeans` mirror the three `kmeans*` entry points
and take two optional numbers: how many passes to make, and how many points per
EMA update.

```js
import { emeans_rgba } from "kmeans-wasm";

const palette = emeans_rgba(image.data, 16);          // 5 passes by default
const quicker = emeans_rgba(image.data, 16, 3);       // fewer passes
const pinned = emeans_rgba(image.data, 16, 5, 4096);  // explicit batch size
```

There is no convergence threshold: the cosine schedule fixes the number of
updates, so `epochs` is the quality/latency dial and `batchSize` only needs
setting when you are reproducing a run.

`emeans(data, k, epochs?, batchSize?)` returns the same object as `kmeans`, with
two differences:

- `it` is the number of completed passes, not Lloyd iterations.
- `idxs` is each point's assignment during the final pass. The centroids move
  during that pass, so an entry need not be the nearest of the returned final
  centroids. Use `result.test(point)` for a nearest-centroid query.

e-means is stochastic. The wasm build draws the initial centroids from
`Math.random` and seeds a local stream from it for the rest of the run, so
patching `Math.random` before a call makes the run reproducible and gives both
clustering families the same starting centroids. The packed paths use the same
distinct-color reduction as `kmeans*`, weighted by pixel count; batches are
filled by total weight, and large weights are split into chunks so one dominant
color does not turn a batch into a block of itself.

Clusters below a tenth of the mean mass are revived every five updates by
splitting a donor cluster, disabled at two dimensions or fewer and when `k`
exceeds the number of distinct points. Finalization replaces each non-empty
centroid with its mean over the last pass; an empty cluster keeps its position.

Five passes is the default. The table below uses it. Three passes are faster but
can lose 10% or more on an individual image; ten passes buy almost nothing over
five.

### Arguments and errors

All entry points throw a `string` on invalid input, so `try`/`catch` and
`String(error)` are enough to report a problem:

- `k` must be at least 2
- `maxIter` must be at least 1
- `epochs` must be at least 1
- `batchSize` must be at least 1
- `convergenceThreshold` must not be negative
- `kmeans_rgb` and `kmeans_rgba` need a length that is a multiple of 3 or 4
- `emeans_rgb` and `emeans_rgba` need the same length multiple
- `kmeans` and `emeans` need every point to have the same dimension
- `apply_palette` needs a component count of 3 or 4, a `pixels` length that is a
  multiple of it, and a non-empty `palette` that is also a multiple of it

`kmeans_rgb` and `kmeans_rgba` default `convergenceThreshold` to `0.1`, below
the resolution of the `u8` result, so a round only runs while it can still change
a color. Pass `0` to always run every iteration. `kmeans` has no such default
because `f64` centroids can move by less than one; it runs the full `maxIter`
unless you set a threshold. The `emeans*` entry points always run the requested
number of passes.

## Benchmarks

The reference tables come from the release WebAssembly build on Node.js 26.10.0
over an AMD Ryzen 5 9600X, measured as the median of five harness runs. Each
harness run is itself a median of eight measured calls after two warm-ups. The
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
| RGB random pixels | 1,000 | 2 | 0.47 ms | 0.41 ms | 0.9× |
| RGB random pixels | 10,000 | 4 | 6.61 ms | 20.32 ms | 3.1× |
| RGB random pixels | 10,000 | 16 | 12.84 ms | 59.23 ms | 4.6× |
| RGB random pixels | 100,000 | 2 | 41.60 ms | 45.62 ms | 1.1× |
| RGB random pixels | 100,000 | 8 | 103.81 ms | 289.67 ms | 2.8× |
| RGB random pixels | 100,000 | 32 | 247.03 ms | 1677.87 ms | 6.8× |

### Reference general vector-space results

Same setup, over points of varying dimension and cluster count.

| Points | Dimensions | Clusters | `kmeans` | `kmeans_rgb` | `skmeans` | Speed-up | Packed speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3 | 2 | 0.62 ms | 0.37 ms | 0.81 ms | 1.3× | 1.7× |
| 10,000 | 3 | 2 | 5.61 ms | 3.73 ms | 6.04 ms | 1.1× | 1.5× |
| 10,000 | 3 | 10 | 10.87 ms | 9.22 ms | 61.59 ms | 5.7× | 1.2× |
| 10,000 | 3 | 50 | 34.78 ms | 29.66 ms | 364.86 ms | 10.5× | 1.2× |
| 100,000 | 3 | 10 | 126.62 ms | 118.03 ms | 1145.68 ms | 9.0× | 1.1× |
| 10,000 | 10 | 10 | 25.35 ms | — | 306.38 ms | 12.1× | — |
| 10,000 | 50 | 10 | 83.15 ms | — | 1456.35 ms | 17.5× | — |
| 10,000 | 50 | 50 | 263.07 ms | — | 4175.00 ms | 15.9× | — |

`Speed-up` is `skmeans` divided by `kmeans`. `Packed speed-up` is `kmeans`
divided by `kmeans_rgb` on the same three-dimensional points. Random points are
99.7% distinct, so the packed path saves the boundary copy rather than the
clustering work.

### Reference real-image results

Every column is a full-image measurement of the same pixels.

| Image | Pixels | Distinct | Colors | `kmeans_rgb` | `kmeans_rgba` | `skmeans` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blue-marble | 922,560 | 6.2% | 8 | 56.11 ms | 56.32 ms | 3905.20 ms | 69.0× |
| blue-marble | 922,560 | 6.2% | 16 | 109.24 ms | 108.40 ms | 11133.62 ms | 101.8× |
| blue-marble | 922,560 | 6.2% | 32 | 154.17 ms | 153.15 ms | 13079.75 ms | 83.7× |
| blue-marble | 922,560 | 6.2% | 64 | 267.45 ms | 241.97 ms | 24919.62 ms | 95.0× |
| city | 307,200 | 45.8% | 8 | 153.04 ms | 151.03 ms | 971.58 ms | 6.0× |
| city | 307,200 | 45.8% | 16 | 211.08 ms | 201.75 ms | 2331.48 ms | 11.2× |
| city | 307,200 | 45.8% | 32 | 336.55 ms | 335.36 ms | 4422.78 ms | 13.1× |
| city | 307,200 | 45.8% | 64 | 657.36 ms | 600.77 ms | 8133.54 ms | 13.2× |
| coast | 307,200 | 30.5% | 8 | 91.60 ms | 93.07 ms | 765.01 ms | 7.5× |
| coast | 307,200 | 30.5% | 16 | 150.72 ms | 147.46 ms | 2358.21 ms | 15.8× |
| coast | 307,200 | 30.5% | 32 | 265.74 ms | 241.57 ms | 4407.00 ms | 17.3× |
| coast | 307,200 | 30.5% | 64 | 474.47 ms | 465.97 ms | 8149.37 ms | 17.2× |
| harbour | 307,200 | 36.6% | 8 | 102.03 ms | 102.34 ms | 800.26 ms | 7.8× |
| harbour | 307,200 | 36.6% | 16 | 142.86 ms | 135.50 ms | 2116.45 ms | 14.7× |
| harbour | 307,200 | 36.6% | 32 | 291.07 ms | 264.26 ms | 4176.57 ms | 14.3× |
| harbour | 307,200 | 36.6% | 64 | 524.26 ms | 485.55 ms | 7993.00 ms | 15.5× |

### Reference e-means results

Lloyd is 100 iterations with no early convergence. e-means is five passes with
the default batch and step laws. Both sides are seeded from the same stream
before every call, so their initial centroids match. MSE is the weighted mean
squared nearest-palette distance over distinct colors, which is also the mean
over pixels.

| Image | Pixels | Colors | Lloyd MSE | e-means MSE | MSE ratio | `kmeans_rgb` | `emeans_rgb` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blue-marble | 922,560 | 8 | 273.75 | 221.57 | 0.809 | 44.98 ms | 13.08 ms | 3.4× |
| blue-marble | 922,560 | 16 | 104.27 | 96.06 | 0.921 | 91.70 ms | 16.79 ms | 5.5× |
| blue-marble | 922,560 | 32 | 56.45 | 50.81 | 0.900 | 124.22 ms | 22.54 ms | 5.5× |
| city | 307,200 | 8 | 958.17 | 958.29 | 1.000 | 132.40 ms | 28.30 ms | 4.7× |
| city | 307,200 | 16 | 497.30 | 485.70 | 0.977 | 174.46 ms | 36.62 ms | 4.8× |
| city | 307,200 | 32 | 270.61 | 267.36 | 0.988 | 340.15 ms | 49.94 ms | 6.8× |
| coast | 307,200 | 8 | 713.37 | 713.27 | 1.000 | 89.46 ms | 17.95 ms | 5.0× |
| coast | 307,200 | 16 | 349.02 | 348.57 | 0.999 | 134.28 ms | 24.70 ms | 5.4× |
| coast | 307,200 | 32 | 227.09 | 205.10 | 0.903 | 222.66 ms | 33.82 ms | 6.6× |
| harbour | 307,200 | 8 | 1063.00 | 1063.04 | 1.000 | 89.41 ms | 23.21 ms | 3.9× |
| harbour | 307,200 | 16 | 492.41 | 492.69 | 1.001 | 154.91 ms | 30.71 ms | 5.0× |
| harbour | 307,200 | 32 | 256.70 | 250.10 | 0.974 | 296.20 ms | 41.12 ms | 7.2× |

Median five-epoch speed-up is about `5.4x`; three epochs are faster, around `7.8x` median.

### Reference mapping results

The baseline is a per-pixel search over the palette, memoised by packed color.
Both columns are byte-identical on every row.

Median of eight measured calls after two warm-ups, one harness run.

| Image | Pixels | Distinct | Colors | `apply_palette` | JavaScript loop | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blue-marble | 922,560 | 6.2% | 8 | 9.75 ms | 19.45 ms | 2.00× |
| blue-marble | 922,560 | 6.2% | 16 | 8.91 ms | 18.83 ms | 2.11× |
| blue-marble | 922,560 | 6.2% | 32 | 11.35 ms | 22.12 ms | 1.95× |
| blue-marble | 922,560 | 6.2% | 64 | 15.63 ms | 26.25 ms | 1.68× |
| city | 307,200 | 45.8% | 8 | 10.66 ms | 14.19 ms | 1.33× |
| city | 307,200 | 45.8% | 16 | 13.12 ms | 18.35 ms | 1.40× |
| city | 307,200 | 45.8% | 32 | 17.30 ms | 23.10 ms | 1.34× |
| city | 307,200 | 45.8% | 64 | 27.12 ms | 33.13 ms | 1.22× |
| coast | 307,200 | 30.5% | 8 | 7.49 ms | 11.05 ms | 1.48× |
| coast | 307,200 | 30.5% | 16 | 9.03 ms | 11.47 ms | 1.27× |
| coast | 307,200 | 30.5% | 32 | 12.00 ms | 15.53 ms | 1.29× |
| coast | 307,200 | 30.5% | 64 | 19.16 ms | 22.15 ms | 1.16× |
| harbour | 307,200 | 36.6% | 8 | 7.98 ms | 9.58 ms | 1.20× |
| harbour | 307,200 | 36.6% | 16 | 10.43 ms | 11.97 ms | 1.15× |
| harbour | 307,200 | 36.6% | 32 | 14.01 ms | 16.56 ms | 1.18× |
| harbour | 307,200 | 36.6% | 64 | 22.52 ms | 25.15 ms | 1.12× |

Above roughly half distinct, a per-pixel search becomes competitive again. A
call on input that does not repeat at all costs about 0.83× because the table
build is wasted before the fallback search.

## Contributing

Pull requests and issues are welcome. Add tests for new features and bug fixes.

## License

[MIT](./LICENSE_MIT)
