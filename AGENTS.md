# AGENTS.md

Working notes for agents and maintainers on `kmeans-wasm`. Read this before
changing anything; it records the things that are expensive to rediscover.

## What the project is

A Hamerly k-means implementation in Rust compiled to WebAssembly, published to
npm as `kmeans-wasm`. Three public entry points:

| Export | Input | Returns | Notes |
| --- | --- | --- | --- |
| `kmeans_rgb` | `Uint8Array`, 3 components per point | `Uint8Array` of centroids | packed color path |
| `kmeans_rgba` | `Uint8Array`, 4 components per point | `Uint8Array` of centroids | clusters alpha too |
| `kmeans` | `Array<Array<number>>` | object with `k`, `it`, `centroids`, `idxs`, `test` | arbitrary dimensions |

`kmeans` returns a JS object, so it is only usable from JavaScript. The `js_sys`
calls abort the process on native targets, which is why the general API has no
native test coverage and its benchmark harness is a Node script.

## Layout

```
src/lib.rs                 public API, argument validation, input conversion
src/kmeans_triangle.rs     the clustering core: hamerly_kmeans and helpers
benches/kmeans_rgb.rs      native Criterion bench, packed 3-component
benches/kmeans_rgba.rs     native Criterion bench, packed 4-component
js_bench/src/rgb.ts        Node harness: kmeans_rgb vs skmeans
js_bench/src/vector.ts     Node harness: kmeans vs skmeans vs kmeans_rgb
docs/                      GitHub Pages playground (index.html, app.js, styles.css)
scripts/                   npm package finalizer, npm staging helper
tests/web.rs               wasm-only tests (browser)
tests/quantize.rs          native tests for the packed paths
```

`docs/wasm/`, `pkg/`, `kmeans-wasm-node/`, `target/` and `node_modules/` are
generated and git-ignored.

## Commands

```sh
npm ci
npm run check          # lint + native tests + package validation
npm run lint           # cargo fmt --check + clippy -D warnings (all targets, host)
npm test               # cargo test --locked
npm run test:web       # wasm-pack test --chrome --headless  (needs Chrome)
npm run build          # wasm-pack -> pkg/ + finalize-npm-package.mjs
npm run pages:build    # wasm-pack --target web -> docs/wasm/
npm run bench:rgb      # rebuilds the node target, then runs js_bench/src/rgb.ts
npm run bench:kmeans   # same for the general API harness
npm run release:dry    # release-it dry run
```

`npm run check` is the gate. CI also runs `test:web`, both benchmark harnesses,
`npm audit`, and greps `pkg/README.md` for the two reference benchmark tables, so
do not rename those headings.

## Toolchain

Rust 1.86 minimum, edition 2024. CI pins 1.96.0. Node `^22.22.2 || ^24.15.0 ||
>=26`, npm 12.1.0 exactly (`devEngines` fails the install otherwise). wasm-pack
0.15.0. WebAssembly SIMD plus bulk-memory, reference-types, sign-ext and
nontrapping float-to-int are enabled through the `wasm-opt` flags in
`Cargo.toml`; the release profile uses `opt-level=3`, LTO, `codegen-units=1`,
`strip` and `panic = "abort"`.

A Rust panic in the wasm build becomes an `unreachable` trap, so a panic shows up
in the browser as `Could not run the comparison: unreachable` with no stack.

## Invariants that must not break

**Clustering output is bit-for-bit stable.** Consumers depend on reproducible
palettes, and the published benchmark tables assume the arithmetic does not
drift. `src/kmeans_triangle.rs` has a `tests` module that fingerprints the
iteration count, every centroid bit pattern and every assignment with FNV-1a
across five shape combinations. If you change the arithmetic on purpose, that is
the test you re-record, and it must be a deliberate, reviewed decision.

Two subtleties that are easy to break:

- The distance accumulation order must stay dimension 0 upward. Reassociating
  the sum changes the last bits.
- The centroid scan must stay in ascending index order with a strict `<`
  comparison, so ties resolve to the lowest index. Pre-seeding the scan with a
  known distance *and* keeping the ascending order is safe; pre-seeding it and
  skipping the current centroid changes tie-breaking.

**Native and wasm initialization share one algorithm, but only the entropy
differs.** wasm must read `js_sys::Math::random`, because that is what callers
replace to make a run reproducible; `docs/app.js` patches it before every
measurement so all three implementations start from the same centroids. Natively
it is a seeded `SmallRng` so tests and benchmarks repeat. Do not unify these into
one implementation without reworking the page's seeding, and keep the native
stream deterministic.

**`kmeans` result shapes are load-bearing.** `centroids` must be a dense array
of `k` arrays of `dimensions` numbers. It was briefly built with
`Array::new_with_length` plus `push`, which produced a sparse array of length
`2k` whose first `k` entries were holes, so every `result.centroids[0]` was
`undefined`. `test` must be called with the result as the receiver, because it
reads `this.centroids`.

**`rand` is a dependency on every target** and needs
`getrandom = { version = "0.4", features = ["wasm_js"] }` for wasm32, or
`getrandom` fails to compile with an explicit `compile_error!`. Keeping `rand` in
the wasm build costs nothing: the artifact measures 33,137 bytes with and
without it, because the wasm path never calls it and LTO drops it.

## Performance work: how to measure here

**Wall clock on this machine is not trustworthy.** An identical binary has been
observed varying by 50% run to run, and two different methods disagreed about the
same code (criterion reported 93.9 ms where a min-of-5 probe reported 57.4 ms for
the same case). Load average is low and the host appears shared. Assume any
single wall-clock number here is wrong unless it is large.

What to use instead, in order of preference:

1. **Deterministic counters.** Allocation counts (a `GlobalAlloc` wrapper) and
   distance-evaluation counts (a counter in `get_distance_squared`, `#[cfg(test)]`
   only) are exact and reproducible. A previous change went from 409,680
   allocations to 48, and from 693,128 to 197,964 distance evaluations, both
   provable without a stopwatch.
2. **In-process interleaved micro-benchmarks.** Put every candidate variant in
   one binary, alternate between them, and report the minimum. Drift then hits
   all variants equally. This is how indexing was chosen over iterators for the
   inner distance loop: the variants were within noise at 3 and 8 components and
   indexing was about 4% ahead at 50.
3. **criterion with `--save-baseline` / `--baseline`** for the committed benches,
   and only for large effects.
4. **Median of five harness runs** for anything that goes into the README. One
   sample is not reproducible to better than roughly 30%.

There is no `perf` and no `valgrind` on this machine, so instruction counts have
to come from a counting allocator or a counter in the code.

Keep changes that are neutral or better. Dismiss anything that measures slower,
even if it is cleaner. When a change is ambiguous, leave the code alone and say
so.

## Optimization ideas not yet done

Roughly in the order they seemed most promising:

- The `kmeans` JS boundary still copies every point from a JS array into the flat
  buffer. That interop cost is visible in the published table as the gap between
  `kmeans` and `kmeans_rgb`, and it dominates for small inputs.
- `move_centers` divides by the cluster count, which is zero for an empty cluster
  and yields a NaN centroid. Empty clusters are possible in principle; the
  rejection sampler reduces but does not eliminate the chance.
- The bounds are stored as distances because Hamerly's update is additive in
  distance space, so the square roots cannot be removed without changing the
  algorithm.
- Nothing is SIMD-inherent. The distance loop is left to LLVM auto-vectorization;
  explicit `f64x2` intrinsics were not tried and would need a non-wasm fallback.

## Testing gaps

- The wasm suite only runs in CI. No Chrome or Chromium binary is available in
  the agent environment, so `npm run test:web` cannot be run locally. Every
  assertion destined for `tests/web.rs` should first be checked against the Node
  build to avoid pushing a red CI.
- Native and wasm have separate `get_centroids` entropy paths, so a native test
  cannot cover the wasm one. A bug in the wasm row-slicing was introduced during
  the flat-buffer rewrite and was invisible to every native test; it was caught
  by running `docs/app.js` under a headless DOM harness. Keep that harness idea
  in mind: shim `document`, `Image` and `getContext`, stub `fetch` for `file:`
  URLs so the wasm-pack web glue can load, then import `docs/app.js` directly.
- `docs/app.js` has no automated coverage. A regression there is only visible on
  the deployed page.
