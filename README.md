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

Three entry points, all sharing the same clustering core. Pick the one that matches
your data layout:

| Export | Input | Use it for |
| --- | --- | --- |
| `kmeans_rgb` | `Uint8Array`, 3 values per point | image and palette work |
| `kmeans_rgba` | `Uint8Array`, 4 values per point | the same, when alpha matters |
| `kmeans` | `Array<Array<number>>` | any other number of dimensions |

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
input and the output, with no intermediate conversion:

```js
import { kmeans_rgba } from "kmeans-wasm";

const context = canvas.getContext("2d", { willReadFrequently: true });
const image = context.getImageData(0, 0, canvas.width, canvas.height);

const palette = kmeans_rgba(image.data, 16, 30, 0.1);

// paint: replace every pixel with its nearest palette entry
const output = context.createImageData(canvas.width, canvas.height);
for (let pixel = 0; pixel < canvas.width * canvas.height; pixel += 1) {
  const source = pixel * 4;
  const nearest = nearestPaletteEntry(palette, image.data, source);
  output.data[source] = palette[nearest];
  output.data[source + 1] = palette[nearest + 1];
  output.data[source + 2] = palette[nearest + 2];
  output.data[source + 3] = palette[nearest + 3];
}
context.putImageData(output, 0, 0);

function nearestPaletteEntry(palette, pixels, offset) {
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let entry = 0; entry < palette.length; entry += 4) {
    let distance = 0;
    for (let channel = 0; channel < 4; channel += 1) {
      const delta = pixels[offset + channel] - palette[entry + channel];
      distance += delta * delta;
    }
    if (distance < bestDistance) {
      bestDistance = distance;
      best = entry;
    }
  }
  return best;
}
```

Run a full working version, including a three-way speed comparison against
`kmeans_rgb` and `skmeans`, at
[vusolapohvistr.github.io/kmeans-wasm](https://vusolapohvistr.github.io/kmeans-wasm/).

### Arguments and errors

All three entry points throw a `string` on invalid input, so `try`/`catch` and
`String(error)` are enough to report a problem:

- `k` must be at least 2
- `maxIter` must be at least 1
- `convergenceThreshold` must not be negative
- `kmeans_rgb` and `kmeans_rgba` need a length that is a multiple of 3 or 4
- `kmeans` needs every point to have the same dimension

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

`skmeans` is a JavaScript implementation taking `number[][]`, so it cannot be run
over a whole photograph in reasonable time. Its column is a fixed 8,000 pixel
subsample and is **not** comparable to the full-image columns; the other two
columns are.

| Image | Pixels | Distinct | Colors | `kmeans_rgb` | `kmeans_rgba` | `skmeans` (8k subsample) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blue-marble | 922,560 | 6.2% | 8 | 57.29 ms | 54.10 ms | 26.96 ms |
| blue-marble | 922,560 | 6.2% | 16 | 112.61 ms | 107.21 ms | 62.46 ms |
| blue-marble | 922,560 | 6.2% | 32 | 160.13 ms | 149.31 ms | 68.14 ms |
| blue-marble | 922,560 | 6.2% | 64 | 269.86 ms | 245.98 ms | 91.19 ms |
| city | 307,200 | 45.8% | 8 | 149.17 ms | 156.91 ms | 16.36 ms |
| city | 307,200 | 45.8% | 16 | 216.09 ms | 210.94 ms | 40.19 ms |
| city | 307,200 | 45.8% | 32 | 336.88 ms | 322.92 ms | 81.59 ms |
| city | 307,200 | 45.8% | 64 | 634.86 ms | 614.19 ms | 127.56 ms |
| coast | 307,200 | 30.5% | 8 | 93.31 ms | 92.63 ms | 18.14 ms |
| coast | 307,200 | 30.5% | 16 | 149.45 ms | 144.17 ms | 47.82 ms |
| coast | 307,200 | 30.5% | 32 | 259.77 ms | 251.69 ms | 106.42 ms |
| coast | 307,200 | 30.5% | 64 | 480.97 ms | 441.11 ms | 143.30 ms |
| harbour | 307,200 | 36.6% | 8 | 104.93 ms | 103.64 ms | 14.05 ms |
| harbour | 307,200 | 36.6% | 16 | 143.16 ms | 137.16 ms | 27.15 ms |
| harbour | 307,200 | 36.6% | 32 | 291.68 ms | 274.22 ms | 71.95 ms |
| harbour | 307,200 | 36.6% | 64 | 506.94 ms | 470.41 ms | 123.64 ms |

Read the `Distinct` column against the timings. `blue-marble` is 6.2% distinct, so
it reaches the clustering as 57,086 weighted values instead of 922,560 pixels and
quantizes a megapixel in 160 ms at 32 colours. `city` is 45.8% distinct, so almost
half its colours are unique and there is much less to collapse: 307,200 pixels take
337 ms at the same 32 colours, more than twice the time for three times the data.

## Contributing

Pull requests and issues are welcome. Add tests for new features and bug fixes.

## License

[MIT](./LICENSE_MIT)
