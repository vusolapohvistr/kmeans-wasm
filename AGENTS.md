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
do not rename the `Reference RGB results` and `Reference general vector-space
results` headings.

## Setting up a checkout

Prerequisites: Rust 1.98 or newer, Node `^22.22.2 || ^24.15.0 || >=26`, npm
12.1.0 exactly, wasm-pack 0.15.0, and Chrome or Chromium for the browser tests.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack --version 0.15.0 --locked
npm ci
```

To preview the playground page locally:

```sh
npm run pages:build
npx serve docs
```

The benchmark harnesses rebuild the Node target of the WebAssembly package before
they run, so `npm run bench:rgb` and `npm run bench:kmeans` always measure the
current working tree. Each harness run is a median of eight measured runs after
two warm-ups, and the README tables are the median of five harness runs, because a
single run on a shared machine is not reproducible to better than roughly 30%.

## Toolchain

Rust 1.98 minimum, edition 2024. CI pins 1.98.1 in both workflows, and the local
default is expected to match, because a different LLVM produces a different
artifact and the deployed bundle comes from CI. Node `^22.22.2 || ^24.15.0 ||
>=26`, npm 12.1.0 exactly (`devEngines` fails the install otherwise). wasm-pack
0.15.0, which bundles binaryen 117. The release profile uses `opt-level=3`, LTO,
`codegen-units=1`, `strip` and `panic = "abort"`.

Only the WebAssembly build is published, so the declared MSRV is the floor for
*building* the package rather than a promise about native builds. It was raised to
1.98 for that reason.

### WebAssembly feature flags

There are two places, and they are not the same thing:

- `.cargo/config.toml` sets rustc `-C target-feature` for `wasm32-unknown-unknown`.
  Only `+simd128` is listed. rustc already enables `bulk-memory`, `multivalue`,
  `mutable-globals`, `nontrapping-fptoint`, `reference-types` and `sign-ext` by
  default on that target, so listing them is a no-op. Verify with
  `rustc --print cfg --target wasm32-unknown-unknown`.
- `Cargo.toml` passes `--enable-*` flags to `wasm-opt`, per wasm-pack profile.
  Only `--enable-simd`, `--enable-bulk-memory` and `--enable-nontrapping-float-to-int`
  are needed. Binaryen 117 has sign-extension and reference types on by default,
  so those two flags were removed; the build still produces a byte-identical
  artifact (md5 `28866c3322ee41328ab582ed7d41e011`, 34,016 bytes) without them.

When changing either list, rebuild and compare the artifact hash. A flag that is
redundant must not change the output; if it does, it was doing something.

To find the minimal `wasm-opt` set, run the binaryen binary from the wasm-pack
cache directly, at `~/.cache/.wasm-pack/wasm-opt-*/bin/wasm-opt`, and bisect. It
is not on `PATH`. It is also the only way to inspect the module: `--print` dumps
the text format so you can count v128 opcodes, and dropping a required flag makes
it fail validation naming the offending instruction, which is how the
`memory.copy` and `i32.trunc_sat_f64_u` requirements were found.

`+relaxed-simd` was evaluated and rejected. It produces a byte-identical artifact,
so LLVM emits no relaxed instructions for this code, and Safari does not support
it. It would only cost browser support. Tail calls, `memory64`, WasmGC, extended
const, wide arithmetic, FP16, atomics and multiple memories were all considered
and none of them apply to a flat k-means loop.

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
`getrandom = { version = "0.4.3", features = ["wasm_js"] }` for wasm32, or
`getrandom` fails to compile with an explicit `compile_error!`. Keeping `rand` in
the wasm build costs nothing: the artifact measures the same size with and
without it, because the wasm path never calls it and LTO drops it.

**`fearless_simd` provides the SIMD inner loop.** It is added as
`{ version = "1.0.0", default-features = false, features = ["libm"] }`. The
default `std` feature drags in `futures-util`, `futures-core`, `futures-task`,
`slab` and `pin-project-lite`; `libm` replaces that whole chain with one crate,
and nothing links it because the kernel only uses add, subtract, multiply and
sqrt, all of which are hardware operations. It needs Rust 1.89, which is below
the declared MSRV.

The kernel is generic over `S: Simd` and `hamerly_kmeans_dispatched` selects the
level once per run with `Level::baseline()` and `dispatch!`. Two mistakes are
recorded here because both were made:

- **Do not put `dispatch!` inside the inner loop.** A per-call level check made
  SIMD look 10 to 28% *slower* than scalar, which is the opposite of the truth.
  The level is chosen once and the token is threaded through.
- **`Level::baseline()`, not `Level::new()`.** The package requires `simd128` at
  compile time, so the level is a compile-time constant and there is nothing to
  detect. `baseline` is also `const` and needs no `std`.

Why the kernel is not just left to LLVM: the scalar form serialises every point
through one accumulator, so the floating-point add latency chain rather than the
arithmetic sets the pace. The proof is that the scalar scan takes the same time
at 3, 4 and 8 components. Two lanes halve that chain and halve the instruction
count per point.

Measured on the shipped wasm target, two modules in one process alternating,
minimum of nine: 1.02x at k=2, 1.07x at k=8, 1.15x at k=16, 1.35x at k=32 with
100,000 pixels, 1.19x at 409,600 pixels. Smaller than a native AVX-512 measurement
suggests, because wasm SIMD is fixed at two f64 lanes. The gain grows with the
cluster count, because that is what makes the inner loop long enough to matter.
Cost is 523 bytes, 33,968 to 34,491.

**Rust 1.98's `algebraic_add` family is not the answer to the same problem, and
was rejected.** It permits reassociation rather than giving you vector registers.
Measured on the shipped target it emits zero SIMD instructions for a
runtime-length inner loop, gives 1.00x at 3 and 4 components, regresses to 0.76x
at 8 components with two accumulators, and only reaches 1.22x at 50 components
with four. It also changes results for non-integer input, because reassociating a
sum of non-exact values is observable. `portable_simd` is still unstable on
1.98.1, so `core::simd` is not an option and `fearless_simd` remains the only
route to explicit SIMD without `unsafe`.

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
- Nothing is SIMD-inherent beyond the two-lane kernel. wasm SIMD is fixed at 128
  bits, so there is no wider width to reach for on the shipped target.
- The SIMD kernel only helps once the inner loop is long enough. `kmeans_rgba` at
  four components does one vector op plus a horizontal add per distance, so the
  remaining cost is the per-pair loop overhead rather than the arithmetic.

## Releasing

Releases are cut locally with [`release-it`](https://github.com/release-it/release-it).
It reads and writes the version in `Cargo.toml`, refreshes `Cargo.lock`, runs the
checks, commits, tags `vX.Y.Z`, pushes, and creates the GitHub Release. Nothing is
published from CI, and there is no npm token in the repository.

1. Merge to `main` and confirm CI is green:

   ```sh
   git switch main
   git pull --ff-only
   npm ci
   npm run release:dry
   ```

2. Start the release and pick the increment. A new public entry point or a
   behavior change is a minor; a pure fix is a patch.

   ```sh
   npm run release
   ```

3. The `after:release` hook runs `npm run release:stage`, which calls
   `npm stage publish --tag <latest|next> ./pkg`. Staging needs no 2FA prompt, but
   a human must inspect the tarball and approve it with 2FA:

   ```sh
   npm stage list kmeans-wasm
   npm stage download <stage-id>
   npm stage approve <stage-id>
   ```

   Reject with `npm stage reject <stage-id>`. Stable versions use the `latest` tag
   and prereleases use `next`.

`GITHUB_TOKEN` is optional. Without it `release-it` opens a prefilled GitHub
Release page for manual confirmation; with a repo-scoped token set, the Release is
created automatically. Do not commit `.npmrc` or tokens. If a token is needed, use
an npm granular access token with **Read and write (stage only)** permissions for
`kmeans-wasm`, and never a bypass-2FA token for direct publishing.

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
