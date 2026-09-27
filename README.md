# kmeans-wasm

A fast k-means clustering implementation written in Rust and compiled to WebAssembly. It supports color quantization and general vector-space data, with JavaScript and TypeScript bindings.

## Credits

This library would not exist without the researchers whose work it builds on. Every
algorithmic idea here is someone else's, reimplemented and tuned.

**The clustering core.** The algorithm is Hamerly k-means, from Grant Hamerly and
Charles Elkan, *Efficient Algorithms for Clustering with Bounds*, *Data Mining and
Knowledge Discovery* 24(3), 2010, doi `10.1007/s10618-009-0516-5`. Each point keeps
a lower bound on the distance to its second-closest centroid, and the centroids' own
movement tightens those bounds by the triangle inequality, so most centroids are
never measured at all. This is the entire reason the implementation is fast.

The loop it accelerates is Stuart Lloyd's, *Least Squares Quantization in PCM*,
*IEEE Transactions on Information Theory* 28(2), 1982, [doi][lloyd]: assign every
point to the nearest centroid, then move each centroid to the mean of its members.
Hamerly's bounds avoid most of the distance measurements in that loop and nothing
else. The direct ancestor is Charles Elkan, *Using the Triangle Inequality to
Accelerate k-Means*, *ICML* 2004, which stores a bound per centroid per point
instead of one per point.

**Alternatives measured and rejected.** Hamerly's bounds are the cheapest in memory
of the stored-bounds family, keeping one bound per point where Elkan keeps one per
centroid per point, and that is what makes them fast rather than fast *and* small.
Two exact algorithms benchmark ahead of it anyway: Christoph Borgelt's *Even Faster
Exact k-Means Clustering*, *IDA* 2020, [doi][borgelt], and Aurélien Newling and
François Fleuret's *Fast k-Means with Accurate Bounds*, *MLG* 2016, in [*PMLR*
v48][newling], which report roughly 1.3x to 3x over Hamerly. A recent idea worth
revisiting is Max Pernklau and Nikita Averitchev,
*Extending k-Means Clustering with Ptolemy's Inequality*, *BTW* 2025, [open
access][ptolemy]: Ptolemy's inequality gives tighter bounds than the triangle
inequality, and the gains grow with the cluster count and as dimension falls, which
is the shape of the packed color workload. None of these were ported, because this
implementation already spends only about 14% of a full distance scan per round, so
the extra bound arithmetic they add plausibly costs more than the distances they
save. The measurements behind that decision are in `AGENTS.md`.

**Seeding.** David Arthur and Sergei Vassilvitskii, *k-means++: the Advantages of
Careful Seeding*, *SODA* 2007, [PDF][arthur], is the standard better seeding
scheme. It was implemented and measured here, and rejected: it improved the median
iteration count on three of four shapes and worsened it on one, all within a few
percent, while costing about 7% per run. Some of the speedup in the current version
comes from not paying for it.

**Color quantization.** M. Emre Celebi, *Improving the Performance of K-Means for
Color Quantization*, [arXiv:1101.0395][celebi], and *Fast Color Quantization Using
Weighted Sort-Means Clustering*, *JOSA A* 26(11), 2009, [doi][celebi2]. The
observation that clustering an image's distinct colors with weights, rather than
every pixel, gives the same result, together with the measured distinct-color
fractions for the standard test images, 7% to 58%. A prototype of that reduction
reached 16x to 78x on flat-region input, where pixels repeat heavily, and was
measured here but not shipped.

**A plausible reference that turned out to be the wrong one.** Tai Dinh, Wong
Hauchi, Philippe Fournier-Viger, Daniil Lisik, Minh-Quyet Ha, Hieu-Chi Dam and
Van-Nam Huynh, *Categorical Data Clustering: 25 Years Beyond k-Modes*,
[arXiv:2408.17244][categorical], accepted at *Expert Systems with Applications*,
is the obvious reference for a library that
takes `u8` values. It is the wrong one. Its dissimilarity measures are nominal,
simple matching above all, which treats a difference of 1 in red and a difference of
1 in blue as equally far. RGB is discrete but still metric, so Euclidean is correct
and Hamming would degrade the palette.

**Not used.** Theodore Elfving and Einar Carlsson, *Efficient Algorithms for Gaussian
Mixture Models in the EM Algorithm*, *SIAM Journal on Matrix Analysis and
Applications* 17(2), 2005, doi `10.1137/S0895479897328291`, describes the
conjugate-gradient step that accelerates Lloyd's iteration. It is the most promising
known way to cut this library's iteration count, and it is not implemented.

[lloyd]: https://doi.org/10.1109/TIT.1982.1056489
[arthur]: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
[celebi]: https://arxiv.org/abs/1101.0395
[celebi2]: https://doi.org/10.1364/JOSAA.26.002434
[borgelt]: https://doi.org/10.1007/978-3-030-44584-3_8
[newling]: https://proceedings.mlr.press/v48/
[ptolemy]: https://dl.gi.de/items/e0507a05-fc07-45fb-b5b8-06e50def5007
[categorical]: https://arxiv.org/abs/2408.17244

## A note on how this was written

From version **3.2.0** onward, most of the work on this library was done by AI
assistants: the SIMD kernel, the allocation and distance-evaluation work, the
color quantization entry point, the benchmark harnesses, the playground, the
regression tests, and the bug fixes. That work is only possible because the
researchers listed above had already done the hard part, and the results in the
tables below are reported as carefully as the tools allowed so the claims can be
checked rather than taken on trust.

The people behind those papers deserve the credit, and the bugs in this library
are the AI's. Please cite them, not this repository, when describing the
algorithm.

## Requirements

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

### Reference RGB results

Median wall times, two warm-ups and eight measured runs per harness run, 100
maximum iterations, and the median of five harness runs on top of that. Ratio
columns are recomputed from the median times rather than averaged.

| Test | Pixels | Colors | `kmeans_rgb` | `skmeans` | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB random pixels | 1,000 | 2 | 0.16 ms | 0.37 ms | 2.3× |
| RGB random pixels | 10,000 | 4 | 0.77 ms | 5.06 ms | 6.6× |
| RGB random pixels | 10,000 | 16 | 1.87 ms | 9.75 ms | 5.2× |
| RGB random pixels | 100,000 | 2 | 4.71 ms | 17.69 ms | 3.8× |
| RGB random pixels | 100,000 | 8 | 11.12 ms | 69.17 ms | 6.2× |
| RGB random pixels | 100,000 | 32 | 29.54 ms | 148.88 ms | 5.0× |

### Reference general vector-space results

Same measurement setup, over points of varying dimension and cluster count.

| Points | Dimensions | Clusters | `kmeans` | `kmeans_rgb` | `skmeans` | Speed-up | Packed speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3 | 2 | 0.36 ms | 0.08 ms | 0.48 ms | 1.3× | 4.5× |
| 10,000 | 3 | 2 | 1.74 ms | 0.53 ms | 2.50 ms | 1.4× | 3.3× |
| 10,000 | 3 | 10 | 2.62 ms | 1.42 ms | 14.15 ms | 5.4× | 1.8× |
| 10,000 | 3 | 50 | 5.53 ms | 4.32 ms | 33.75 ms | 6.1× | 1.3× |
| 100,000 | 3 | 10 | 26.60 ms | 12.58 ms | 149.58 ms | 5.6× | 2.1× |
| 10,000 | 10 | 10 | 5.33 ms | — | 23.00 ms | 4.3× | — |
| 10,000 | 50 | 10 | 20.04 ms | — | 64.23 ms | 3.2× | — |
| 10,000 | 50 | 50 | 25.93 ms | — | 116.08 ms | 4.5× | — |

`Speed-up` is `skmeans` divided by `kmeans`. `Packed speed-up` is `kmeans` divided by `kmeans_rgb` on the same three-dimensional points, and is shown only where the packed three-component API applies.

The general call has to copy every point across the JavaScript boundary, which is
a fixed cost per point. It dominates the small rows and shrinks as the cluster
count grows, which is why `kmeans_rgb` and `kmeans_rgba` are worth reaching for
whenever the data is three or four values wide.

One caveat about these tables: the harness generates uniform random pixels, which
are about 99% distinct. Real images repeat heavily, and the flat regions that
canvas, PNG, screenshots and icons produce can be under 1% distinct. Anything
that exploits repetition will look worse here than it is in practice, and
`kmeans_rgb` does not exploit it at all. See the credits above for the reduction
that would, and why it is not shipped.

## Contributing

Pull requests and issues are welcome. Add tests for new features and bug fixes.

## License

[MIT](./LICENSE_MIT)
