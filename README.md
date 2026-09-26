# kmeans-wasm

A fast k-means clustering implementation written in Rust and compiled to WebAssembly. It supports color quantization and general vector-space data, with JavaScript and TypeScript bindings.

Version 3 uses WebAssembly SIMD (`simd128`) for performance. Use a runtime with SIMD support, such as Chrome 91+, Firefox 89+, or Safari 16.4+.

Everything else the module relies on, including bulk memory, reference types, sign extension, non-trapping float-to-int conversion and multi-value, has been on by default in every major browser for several years and is assumed rather than negotiated.

## Features

- Hamerly k-means algorithm
- RGB and RGBA color quantization
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

A `convergenceThreshold` of `0` runs until `maxIter`. Passing a small positive
value such as `0.1` stops earlier once the centroids stop moving, which is what
you normally want for interactive work.

## Benchmarks

Both reference tables below come from the release WebAssembly build, measured
locally on Node.js 26.10.0 over an AMD Ryzen 5 9600X, as the median of five runs
of the harness. The distance kernel is SIMD accelerated, so these numbers
include that. Results vary by CPU, runtime, initialization, and convergence
behavior, and the smallest rows are the noisiest because fixed overhead is a
large share of them.

### Reference RGB results

Median wall times, two warm-ups and eight measured runs per harness run, 100
maximum iterations.

| Test | Pixels | Colors | `kmeans_rgb` | `skmeans` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB random pixels | 1,000 | 2 | 0.15 ms | 0.26 ms | 1.7× |
| RGB random pixels | 10,000 | 4 | 0.73 ms | 3.83 ms | 5.3× |
| RGB random pixels | 10,000 | 16 | 2.24 ms | 9.34 ms | 3.7× |
| RGB random pixels | 100,000 | 2 | 4.23 ms | 16.26 ms | 3.6× |
| RGB random pixels | 100,000 | 8 | 12.66 ms | 66.60 ms | 4.4× |
| RGB random pixels | 100,000 | 32 | 45.62 ms | 128.36 ms | 2.8× |

### Reference general vector-space results

Same measurement setup, over points of varying dimension and cluster count.

| Points | Dimensions | Clusters | `kmeans` | `kmeans_rgb` | `skmeans` | Speed-up | Packed speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3 | 2 | 0.36 ms | 0.07 ms | 0.39 ms | 1.1× | 5.0× |
| 10,000 | 3 | 2 | 1.67 ms | 0.43 ms | 2.52 ms | 1.6× | 3.9× |
| 10,000 | 3 | 10 | 2.69 ms | 1.23 ms | 14.63 ms | 5.6× | 2.2× |
| 10,000 | 3 | 50 | 7.74 ms | 6.71 ms | 33.65 ms | 4.3× | 1.2× |
| 100,000 | 3 | 10 | 26.21 ms | 14.49 ms | 131.60 ms | 4.7× | 1.8× |
| 10,000 | 10 | 10 | 5.67 ms | — | 18.87 ms | 3.5× | — |
| 10,000 | 50 | 10 | 19.05 ms | — | 62.19 ms | 3.1× | — |
| 10,000 | 50 | 50 | 29.66 ms | — | 112.22 ms | 3.8× | — |

`Speed-up` is `skmeans` divided by `kmeans`. `Packed speed-up` is `kmeans` divided by `kmeans_rgb` on the same three-dimensional points, and is shown only where the packed three-component API applies.

The general call has to copy every point across the JavaScript boundary, which is
a fixed cost per point. It dominates the small rows and shrinks as the cluster
count grows, which is why `kmeans_rgb` and `kmeans_rgba` are worth reaching for
whenever the data is three or four values wide.

## Comparison with skmeans

You can compare both libraries at <https://ycatbink0t.github.io/kmeans-web-comparison/>.

## Contributing

Pull requests and issues are welcome. Add tests for new features and bug fixes.

## License

[MIT](./LICENSE_MIT)
