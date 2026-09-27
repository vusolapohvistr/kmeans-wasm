# kmeans-wasm

A fast k-means clustering implementation written in Rust and compiled to WebAssembly. It supports color quantization and general vector-space data, with JavaScript and TypeScript bindings.

## Credits

The clustering core is Hamerly k-means, from Grant Hamerly and Charles Elkan,
*Efficient Algorithms for Clustering with Bounds*, *Data Mining and Knowledge
Discovery* 24(3), 581-606, 2010. It accelerates the iteration Stuart Lloyd
described in *Least Squares Quantization in PCM*, *IEEE Transactions on
Information Theory* 28(2), 129-137, 1982, [doi][lloyd].

Reducing packed input to its distinct colors before clustering follows M. Emre
Celebi, *Fast Color Quantization Using Weighted Sort-Means Clustering*, *JOSA A*
26(11), 2009, [doi][celebi2], and *Improving the Performance of K-Means for Color
Quantization*, [arXiv:1101.0395][celebi].

[lloyd]: https://doi.org/10.1109/TIT.1982.1056489
[celebi]: https://arxiv.org/abs/1101.0395
[celebi2]: https://doi.org/10.1364/JOSAA.26.002434

## A note on how this was written

From version **3.2.0** onward, most of the work on this library was done by AI
assistants: the SIMD kernel, the allocation and distance-evaluation work, the
color quantization entry point, the benchmark harnesses, the playground, the
regression tests, and the bug fixes.

That work is only possible because the researchers credited above had already done
the hard part. Please cite them, not this repository, when describing the
algorithm. The bugs in this library are the AI's.

## Requirements

Version 3 uses WebAssembly SIMD (`simd128`) for performance. Use a runtime with SIMD support, such as Chrome 91+, Firefox 89+, or Safari 16.4+.

Everything else the module relies on, including bulk memory, reference types, sign extension, non-trapping float-to-int conversion and multi-value, has been on by default in every major browser for several years and is assumed rather than negotiated.

## Features

- Hamerly k-means algorithm
- RGB and RGBA color quantization
- Reduces packed input to its distinct colors, weighted by how often each occurs, so real images cluster a point set up to 14x smaller than the pixel count
- Maps a palette back onto an image, also keyed on distinct colors rather than pixels
- Arbitrary numeric vector spaces
- SIMD accelerated inner loop, with no `unsafe` and no runtime feature detection
- JavaScript and TypeScript bindings
- ES module package with a documented `exports` entry point

## Installation

```sh
npm install kmeans-wasm
```

The published package targets JavaScript bundlers and exposes an ES module.

## Usage

Three clustering entry points, all sharing the same clustering core, plus a
mapping entry point that turns a palette back into an image. Pick the one that
matches your data layout:

| Export | Input | Use it for |
| --- | --- | --- |
| `kmeans_rgb` | `Uint8Array`, 3 values per point | image and palette work |
| `kmeans_rgba` | `Uint8Array`, 4 values per point | the same, when alpha matters |
| `kmeans` | `Array<Array<number>>` | any other number of dimensions |
| `apply_palette` | `Uint8Array`, a palette | replacing each pixel with its nearest entry |

The packed entry points take a flat typed array and are several times faster than
the general one, because nothing has to be copied across the JavaScript boundary
per point. The general entry point is the only option once you have more or fewer
than three or four values per point.

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

`kmeans_rgb` returns a `Uint8Array` containing the RGB centroids. The input length
must be a multiple of three, one value per channel per pixel.

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

`kmeans_rgba` mirrors `kmeans_rgb` for four-component vectors and returns a `Uint8Array` containing the RGBA centroids. It is the drop-in choice for `ImageData.data` and other buffers that interleave red, green, blue, and alpha, because no repacking is needed before clustering. The alpha channel is clustered like any other component, so the function also works for data that mixes transparent and opaque pixels. The input length must be a multiple of four.

### Quantizing an image in the browser

Because `ImageData.data` is already a packed RGBA buffer, the canvas is both the
input and the output, with no intermediate conversion. `apply_palette` does the
mapping half, and returns RGBA, which is what `putImageData` takes:

```js
import { kmeans_rgba, apply_palette } from "kmeans-wasm";

const context = canvas.getContext("2d", { willReadFrequently: true });
const image = context.getImageData(0, 0, canvas.width, canvas.height);

const palette = kmeans_rgba(image.data, 16, 30, 0.1);

const output = context.createImageData(canvas.width, canvas.height);
output.data.set(apply_palette(image.data, palette, 4));
context.putImageData(output, 0, 0);
```

`apply_palette` works by resolving every distinct color to its nearest palette
entry once, then mapping each pixel with a single table probe, so the cost is
proportional to the number of *distinct* colors rather than to pixels times
entries. On the four photographs in `js_bench/images` that is 1.24x to 2.00x
faster than searching the palette per pixel in JavaScript, and the result is
byte-identical. A tie goes to the lower palette index, matching the clustering
core, so a hand-written search with a strict `<` produces the same bytes.

The same function handles the mapping for either stride: pass `3` with
`kmeans_rgb` and it writes an opaque alpha for you, since a three-component
palette has none to carry.

Run a full working version, including a three-way speed comparison against
`kmeans_rgb` and `skmeans`, at
[vusolapohvistr.github.io/kmeans-wasm](https://vusolapohvistr.github.io/kmeans-wasm/).

### Arguments and errors

All entry points throw a `string` on invalid input, so `try`/`catch` and
`String(error)` are enough to report a problem:

- `k` must be at least 2
- `maxIter` must be at least 1
- `convergenceThreshold` must not be negative
- `kmeans_rgb` and `kmeans_rgba` need a length that is a multiple of 3 or 4
- `kmeans` needs every point to have the same dimension
- `apply_palette` needs a component count of 3 or 4, a `pixels` length that is a
  multiple of it, and a non-empty `palette` that is also a multiple of it

`kmeans_rgb` and `kmeans_rgba` default `convergenceThreshold` to `0.1`,
which is below the resolution of the `u8` result, so a round is only worth
running while it can still change a colour. On flat-region imagery the palette
comes back byte-identical after two rounds instead of the full `maxIter`. Pass
`0` to always run every iteration. `kmeans` has no such default: it returns
`f64` centroids, where sub-unit precision is meaningful, so it runs the full
`maxIter` unless you set `convergenceThreshold` yourself.

## Benchmarks

Both reference tables below come from the release WebAssembly build, measured
locally on Node.js 26.10.0 over an AMD Ryzen 5 9600X, as the median of five runs
of the harness. The distance kernel is SIMD accelerated, so these numbers
include that. Results vary by CPU, runtime, initialization, and convergence
behavior, and the smallest rows are the noisiest because fixed overhead is a
large share of them.

`skmeans` has no convergence threshold: it runs until no point changes cluster or
the iteration cap is hit. The only stopping rule both libraries can share is no
early exit at all, so that is what these tables use, and it is the conservative
direction — `kmeans_rgb` then does strictly more work than `skmeans`, which stops
as soon as the assignments settle. The speed-ups are therefore a floor. In normal
use the packed entry points default to a threshold of `0.1` and are faster still;
`skmeans` cannot be given the same setting, so it is not measured that way here.

### Reference RGB results

Median wall times, two warm-ups and eight measured runs per harness run, 100
maximum iterations, and the median of five harness runs on top of that. Ratio
columns are recomputed from the median times rather than averaged.

| Test | Pixels | Colors | `kmeans_rgb` | `skmeans` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB random pixels | 1,000 | 2 | 0.47 ms | 0.41 ms | 0.9× |
| RGB random pixels | 10,000 | 4 | 6.61 ms | 20.32 ms | 3.1× |
| RGB random pixels | 10,000 | 16 | 12.84 ms | 59.23 ms | 4.6× |
| RGB random pixels | 100,000 | 2 | 41.60 ms | 45.62 ms | 1.1× |
| RGB random pixels | 100,000 | 8 | 103.81 ms | 289.67 ms | 2.8× |
| RGB random pixels | 100,000 | 32 | 247.03 ms | 1677.87 ms | 6.8× |

### Reference general vector-space results

Same measurement setup, over points of varying dimension and cluster count.

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

`Speed-up` is `skmeans` divided by `kmeans`. `Packed speed-up` is `kmeans` divided by `kmeans_rgb` on the same three-dimensional points, and is shown only where the packed three-component API applies.

The general call has to copy every point across the JavaScript boundary, which is
a fixed cost per point. That is the whole of the `Packed speed-up` column, and it is
small on this data: these are random points, so the clustering itself does real work
and the copy is a tenth of the total. The saving is proportionally much larger when
the clustering is cheap relative to the data, which is the small-cluster and
compressible case the real-image table below covers.

Note the shape of the speed-up column: it is worst at the smallest cluster counts
and best at the largest. `skmeans` scans every centroid for every point on every
round, so it pays `n * k` regardless, while the bounds used here skip most of
those measurements, and the saving grows with `k`. The two rows where `skmeans`
is level or marginally ahead are `k=2`, where there is almost nothing to skip.

### Reference real-image results

The two tables above generate uniform random pixels, which are 99.7% distinct.
Real images are nothing like that: the four below are 6% to 46% distinct. The packed
entry points reduce the input to its distinct colors before clustering, so the
random-pixel tables are close to the worst case for them and these are the
representative ones.

Every column below is a full-image measurement of the same pixels, so they can be
compared directly. `skmeans` gets fewer repeats, one measured run after one warm-up
rather than eight after two, because it costs seconds per run on a megapixel where
this library costs milliseconds; the sample is seconds long, so the relative noise
is much lower than the absolute figures suggest.

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

Read the `Distinct` column against the timings. `blue-marble` is 6.2% distinct, so
it reaches the clustering as 57,086 weighted values rather than 922,560 pixels, and
`skmeans` — which does the full `n * k` scan every round — takes 13 seconds where
this takes 154 ms. `city` is 45.8% distinct, so there is far less to collapse: at
32 colours its 307,200 pixels take 337 ms, more than twice as long as three times
the pixel count, and the speed-up falls from 84x to 13x. The ratio tracks how
compressible the image is, which is the whole mechanism.

### Reference mapping results

Clustering gives you the palette; turning that palette back into an image is the
other half, and it is a per-pixel nearest-colour search. `apply_palette` does it by
resolving every *distinct* colour to its nearest entry once and then mapping each
pixel with a single table probe, so the work follows the number of distinct colours
rather than pixels times entries. Real images repeat heavily, which is what makes
that pay.

The baseline is the loop this replaces: a per-pixel search over the palette,
memoised in a `Map` keyed on the packed colour, which is what a caller had to write
before. Both columns were checked to be **byte-identical on every row**, so these
are speed-ups on identical output rather than on two things that look similar.

Median of eight measured runs after two warm-ups, one harness run, 100 maximum
iterations. That is the same per-measurement protocol as the tables above, but a
single harness run rather than the median of five, so treat these as indicative
rather than as reference figures.

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

The trend is the mechanism again: `blue-marble` is 6.2% distinct and gets the most,
`harbour` at 36.6% gets the least. There is a crossover above roughly half
distinct, where a per-pixel search becomes competitive again because the table has
less to compress, and an `apply_palette` call on input that does not repeat at all
costs 0.83× — the build that was attempted, plus the search it fell back to. That
bound is small, and the real photographs are nowhere near it.

## Contributing

Pull requests and issues are welcome. Add tests for new features and bug fixes.

## License

[MIT](./LICENSE_MIT)
