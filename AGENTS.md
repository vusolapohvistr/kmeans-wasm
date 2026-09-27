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
| `apply_palette` | `Uint8Array`, a palette | `Uint8Array` of RGBA pixels | maps a palette onto an image |

`kmeans` returns a JS object, so it is only usable from JavaScript. The `js_sys`
calls abort the process on native targets, which is why the general API has no
native test coverage and its benchmark harness is a Node script.

## Layout

```
src/lib.rs                 public API, argument validation, input conversion
src/kmeans_triangle.rs     the clustering core: hamerly_kmeans and helpers
src/packed_histogram.rs    collapse to distinct colours, and the palette map
benches/kmeans_rgb.rs      native Criterion bench, packed 3-component
benches/kmeans_rgba.rs     native Criterion bench, packed 4-component
benches/dedup_probe.rs     where the reduction to distinct colours starts paying
benches/mapping_probe.rs   where the palette map beats a per-pixel search
benches/work_split.rs     how much of a round is distances and how much is bounds
js_bench/src/rgb.ts        Node harness: kmeans_rgb vs skmeans
js_bench/src/vector.ts     Node harness: kmeans vs skmeans vs kmeans_rgb
js_bench/src/images.ts     Node harness: the four test photographs
docs/                      GitHub Pages playground (index.html, app.js, styles.css)
scripts/                   npm package finalizer, npm staging helper
scripts/check-page.mjs     runs docs/app.js headless and checks what it painted
scripts/run-web-tests.mjs  finds or fetches a browser and driver for test:web
scripts/probe-gpu-mapping.mjs  checks a WGSL mapping kernel against apply_palette
scripts/probe-wgsl-precision.mjs  asks whether WGSL can match the f64 core
tests/web.rs               wasm-only tests (browser)
tests/quantize.rs          native tests for the packed paths
tests/histogram.rs         native tests for the reduction
tests/palette.rs           native tests for the mapping
```

`docs/wasm/`, `pkg/`, `kmeans-wasm-node/`, `target/` and `node_modules/` are
generated and git-ignored.

## Commands

```sh
npm ci
npm run check          # lint + native tests + package validation
npm run lint           # cargo fmt --check + clippy -D warnings (all targets, host)
npm test               # cargo test --locked
npm run test:web       # the wasm suite, in a real browser
npm run build          # wasm-pack -> pkg/ + finalize-npm-package.mjs
npm run pages:build    # wasm-pack --target web -> docs/wasm/
npm run bench:rgb      # rebuilds the node target, then runs js_bench/src/rgb.ts
npm run bench:kmeans   # same for the general API harness
npm run test:page      # runs docs/app.js headless and checks what it painted
npm run release:dry    # release-it dry run
```

`npm run check` is the gate. CI also runs `test:web`, both benchmark harnesses,
`npm audit`, and greps `pkg/README.md` for the two reference benchmark tables, so
do not rename the `Reference RGB results` and `Reference general vector-space
results` headings.

## Setting up a checkout

Prerequisites: Rust 1.98 or newer, Node `^22.22.2 || ^24.15.0 || >=26`, npm
12.1.0 exactly, and wasm-pack 0.15.0. A browser is needed only for
`npm run test:web`; `npx playwright install chromium` provides one, and
`scripts/run-web-tests.mjs` fetches the chromedriver to match it.

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

`npm run test:web` needs a browser and a chromedriver of the same version; the
script finds or fetches both. On a machine without Chromium's system libraries,
point `BROWSER_LIBS` at a directory holding them.

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
  artifact (md5 `28866c3322ee41328ab582ed7d41e011`, 34,016 bytes at the time)
  without them. The current artifact is larger and is expected to be: the
  unreleased `apply_palette` took it to 41,270 bytes, md5
  `5d1628a94991c3910a36fe4489629f8c`, which is 7,254 bytes for a whole new entry
  point. A caller who never maps pays that in download size, since the wasm
  module is one file and cannot be tree-shaken. The hash is only comparable
  against another build of the same commit, so treat it as a fingerprint rather
  than a target.

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
- The bounds are stored as distances because Hamerly's update is additive in
  distance space, so the square roots cannot be removed without changing the
  algorithm.
- Nothing is SIMD-inherent beyond the two-lane kernel. wasm SIMD is fixed at 128
  bits, so there is no wider width to reach for on the shipped target.
- The SIMD kernel only helps once the inner loop is long enough. `kmeans_rgba` at
  four components does one vector op plus a horizontal add per distance, so the
  remaining cost is the per-pair loop overhead rather than the arithmetic.

## What the literature survey settled, and what it did not

A round of reading turned up two results that change what is worth attempting, and
one that closes off a direction. Recorded so the work is not repeated.

**Hamerly is not the fastest exact algorithm, and that does not matter here.** The
cover-tree paper (arXiv 2410.15117) benchmarks the whole stored-bounds family and
finds Hamerly the slowest of them; Borgelt's *Even Faster Exact k-Means*
(IDA 2020) and Newling and Fleuret's *Fast k-Means with Accurate Bounds* (MLG 2016)
report 1.4x to 3x over it. Measured here, Hamerly already costs only **14% of a full
distance scan per round** at d=3 and 21% at d=4. Those papers reduce distance
*evaluations*, and this crate spent a release making evaluations cheap with SIMD, so
their extra bound arithmetic plausibly costs more than the distances it saves. Their
published wins were measured against a slower kernel. Prototype before porting.
Pernklau and Averitchev's Ptolemy bounds (BTW 2025) are the one to try first if any
of them is: their gains grow with k and as dimension falls, which is our shape.

**k-means++ is not worth it, and that was measured rather than assumed.** Implemented
properly, seeded with weights so a collapsed input would draw the same
distribution. Over five instances per shape it improved median iterations on three
shapes and worsened one, all within a few percent, while costing about 7% per run.
A single run had suggested 1.9x, which was one unlucky draw. The same conclusion
appears in Celebi's initialization comparison, where Forgy lands within a few percent
of the best scheme after refinement.

**The categorical-clustering literature is the wrong reference.** Dinh et al.'s
survey (arXiv 2408.17244) is the obvious place to look for a library taking `u8`
values, and it is a dead end: k-modes and its descendants use nominal dissimilarity
such as simple matching, which rates a difference of 1 in red and a difference of 1
in blue as equally far. RGB is discrete but still metric.

**The one large win, now shipped.** Celebi's weighted sort-means (arXiv 1101.0395)
reduces the input to its distinct values with weights, which is exact because equal
points are interchangeable. `src/packed_histogram.rs` does this for the packed entry
points. Two things were needed to make it trustworthy, and both took a measurement
first:

- **The seed draw has to be weight-proportional, or the reduction quietly changes
  quality.** Drawing uniformly from the distinct values treats a colour seen once
  as likely as one seen ten thousand times. Sampling by weight through an inverse
  CDF on a cumulative table makes the distribution *identical* to drawing pixels
  uniformly, which is what the uncollapsed path does.
- **The exactness is bit-for-bit, and the reason is worth remembering.** The core
  sums `w * v` once per distinct value here and `1 * v` once per pixel otherwise, so
  the two accumulate in different orders, which normally changes the last bits. It
  cannot here: every `u8` value and every partial sum is a small integer, far below
  the 2^53 where `f64` stops being exact, and integer addition is associative. So no
  fingerprint re-record is needed. This does **not** extend to `f64` input, and the
  general `kmeans` API has nothing to collapse anyway.

Measured on 480,000 pixels: 4.9x at 25% distinct, 12.7x at 10%, 50x at 1%. The
bail-out on incompressible input is 0.99x to 1.00x.

**A sampling pre-check cannot decide this, and the reason generalises.** An earlier
version sampled 4096 evenly spaced pixels to guess the duplicate share before
building the table. It reported *zero* duplicates on input that was genuinely 75%
duplicate, because the input repeated with a period far longer than the sample
stride and every sample landed on a different value. No sampling scheme catches
periodic input, so the decision is made from the distinct count alone, with a table
that never grows. The table size is set by the real photographs, not by taste: at
2^17 the reduction declined two of the four test images that clearly benefited.

## Mapping a palette back onto an image, and the predictor that does not work

`apply_palette` is the other half of quantization: clustering gives the palette,
this replaces each pixel with its nearest entry. Callers were writing that loop
themselves, and a per-pixel search is `n * k`, so 29 million distance
evaluations for a megapixel at 32 colours. `src/packed_histogram.rs` resolves
each distinct colour once and then maps every pixel with one table probe, sharing
the open-addressing table that `collapse` already builds.

**The result is bit-identical to a per-pixel search, and that is a property
rather than a hope.** The comparison is strict `<`, so a tie keeps the lower
index, which is what the clustering core does and what a caller's `<` in
JavaScript does. The distance is integer, which is not a shortcut: a channel
difference is at most 255, its square at most 65025, four of those sum to 260100,
so `u32` reproduces the `f64` a JavaScript caller computes exactly. Verified from
20 random starting points and across 80 shape combinations, plus byte-identical
against a transcribed JavaScript loop on all four real images.

**The table wins below about half distinct and loses above it, and the losing
side is bounded rather than open-ended.** `benches/mapping_probe.rs` measures
5.9x at 0.02% distinct, 2.5x at 11%, 1.5x at 32%, level at 46%, 0.8x at 64%.
Above the table's capacity `Table::build` gives up and the call becomes a plain
search, which costs 0.83x: the wasted build plus the search. Every real image is
under the crossover, and through wasm against the JavaScript loop this replaces
it is 1.24x to 2.00x faster with identical bytes.

**A rule that gave up on the *running* distinct share was written, measured, and
removed. Do not write it again.** The share of distinct values seen so far is
tempting: it is free, it is already computed, and it looks like it predicts the
final share. It does not, and the reason is that it is not even monotone. It
*rises* while new colours are still arriving faster than the running average and
only falls afterwards, so it is a hump, not a bound. Measured on the four real
images, the share at the first eighth of the buffer against the final share:

| image | at 1/8 | at 1/4 | at 1/2 | final |
| --- | --- | --- | --- | --- |
| blue-marble | 12.6% | 13.8% | 10.8% | 6.2% |
| city | 34.4% | 46.6% | 48.2% | 45.8% |
| coast | 30.4% | 41.3% | 46.3% | 30.5% |
| harbour | 14.1% | 17.0% | 35.9% | 36.6% |

It over-estimates blue-marble by 2x, under-estimates harbour by 2.5x, and is
level on coast. Any threshold misfires on real photographs. Worse, on a
round-robin generator it misfires catastrophically: `index % distinct` reaches
every colour within the first `distinct` pixels, so the share is at its final
value immediately and a third-share rule abandoned input that was 10.7% distinct
and should have run 2.5x. That is the same class of bug as the sampling pre-check
recorded above, and the same lesson: **a decision needs a signal that is valid
for the input it will actually see, and round-robin is not an image.**

**The probe's generator needed three separate fixes, each of which silently
measured the wrong thing.** It now asserts the distinct count it produced, which
is the only reason the last two were caught. Worth keeping in mind when a
benchmark reports a suspiciously uniform result: at one point the table measured
6.3x at *every* distinct count including 100%, which is impossible, and the
cause was a block size that capped the real distinct count at 300. A second
version fixed the block at 32 by 32 without deriving the height from the requested
count, and capped it again. Random `u8` triples collide by birthday, so a
4096-colour palette was really 4094. All three were invisible in the output and
obvious in hindsight.

## Convergence is data-scale dependent, and the packed default now accounts for it

`total_squared_distance_moved` is compared against an absolute threshold, so whether
that threshold is ever met depends on the scale of the input. With realistic colour
data it effectively never is: runs exhausted all 100 iterations at every k tested.
That was partly an artefact of the NaN bug, since a NaN comparison is always false.

The packed entry points now default to `0.1`, below the resolution of the `u8`
result, and `kmeans` still defaults to `0.0`. Two things to keep in mind:

- **0.1 is small and constant on purpose.** The threshold scales with `k` while the
  output does not. A `k / 4` threshold was measured stopping *before doing any work*
  on a flat image and shifting the palette by 254 levels.
- **Any comparison of two runs must pin the entropy.** The wasm path reads
  `Math.random`, so an unpinned A/B compares two different initialisations. An early
  version of the measurement above appeared to show a 40-level palette regression
  that was entirely two different seeds.

## Measurement traps found the hard way

- **Wall clock on this machine is not trustworthy**, and neither is a single sample
  of anything. An identical binary has been observed varying by 50% run to run.
- **A test that passes with the fix reverted is worse than no test.** An
  end-to-end assertion that the seeds are distinct values passed with the fix
  removed, because with four colours the collision chance is under one in a thousand
  and the native RNG is seeded, so it simply got lucky. The property has to be
  asserted on the draw itself, across a range where a violation is near certain.
- **Check the mechanism before blaming the change.** A `-100%` inertia figure in the
  dedup prototype looked like the new path failing; it was the opposite, the
  collapsed path reaching zero and the full path getting stuck. Print absolute
  values, not percentages of a possibly-tiny reference.
- **`harness = false` plus `test = true` on a bench makes `cargo test` run the whole
  benchmark in a debug build.** It looks exactly like a hang.
- **A comparison table is only worth having if its columns are comparable.** The
  real-image harness originally reported `skmeans` on an 8,000 pixel subsample
  beside two full-image columns, with a footnote admitting the subsample was not
  comparable. That is worse than no column: it invites a comparison that does not
  hold. `skmeans` turns out to run the whole megapixel in 2 seconds at k=8 and 12
  at k=32, so it is now measured on the same pixels with its own, smaller repeat
  count, stated in the table.
- **A benchmark's "random" data has to be checked.** Both harnesses generated pixels
  from the low byte of a 32-bit LCG, whose low bits have a period of 256, so the
  tables were measuring a 256-colour repeating pattern while labelled random. Nothing
  looked wrong for years. It surfaced only when the reduction to distinct colours
  started working and reported a 187x speed-up on "random" input, which is the tell.
  The generator now takes the top bits of a mixed value and the input really is
  99.7% distinct. **An implausible result is evidence about the measurement, not
  about the code.**
- **A test generator needs the same scrutiny.** The collapse test helper re-randomised
  colour per pixel while claiming to build flat regions, so nothing repeated and the
  test asserted the reduction should work on input with no duplicates. A second one
  built a palette with random triples, which collide by birthday, so an exact
  assertion on the distinct count was impossible. Both were the measurement being
  wrong, not the code.

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

## WebGPU: the mapping is feasible and not worth doing, and both halves are now measured

`scripts/probe-gpu-mapping.mjs` runs a WGSL compute kernel that maps a palette
onto an image and checks it against `apply_palette`. Two questions, and the
second one settles it.

**A GPU mapping kernel can be byte-identical, and is.** On all four photographs at
k=32, every one of 1,844,960 pixels matches `apply_palette` exactly. It needs a
strict `<` scanning entries in ascending order, so a tie lands on the lower
entry, and integer arithmetic, which is exact for `u8` input for the same reason
the clustering reduction is. So the correctness bar is met, and it was never the
interesting question.

**The ceiling is 1.05x to 1.10x, because mapping is only 5% to 9% of the work.**
That is the finding, and it is a consequence of `apply_palette` existing. Mapping
is 12 to 16 ms against 135 to 223 ms of clustering, so making mapping *free* buys
almost nothing. Before the reduction and the mapping existed, mapping was the
obvious target; it no longer is. For this to be worth shipping it would have to be
worth more than 9% of the pipeline, and it is not.

**Clustering on the GPU is still the large prize, and it is still blocked.** It is
85% to 95% of the time. Two reasons, both structural. There is no grid-wide
barrier in WebGPU (gpuweb#862), so every Hamerly round needs at least two
dispatches and a host round trip: 96 rounds on blue-marble is 192 dispatches,
about 5.8 ms of pure overhead on Chrome and 199 ms on Firefox at the 1038.7 μs
dispatch cost measured in arXiv 2604.02344, which is more than the entire CPU run.
And the reductions reorder the sums, so the centroids stop being bit-identical,
which breaks the one invariant that matters most here.

**Speed cannot be measured in this environment, and a probe that says otherwise is
wrong.** There is no hardware adapter: `requestAdapter` returns null without
`--enable-unsafe-webgpu`, and with it the adapter is always SwiftShader, at
3.7 to 4.8 ms for a million trivial invocations against roughly 15 μs for real
hardware. The `gpu_ms` column in the probe is a floor, and the script says so.

**Two harness bugs here are worth remembering, because both looked like the code
under test being broken.** A `u32`-per-pixel buffer is *not* a tightly packed RGB
byte stream: passing one to a 3-component entry point reads every pixel three
bytes out of step, which produced 93% mismatches against a correct reference and
read exactly like a broken GPU kernel. And a buffer created with `UNIFORM` but not
`COPY_DST` fails its `writeBuffer` silently, leaving the count at zero, so the
kernel's guard returns before writing and the readback is all zeros. The first was
mine in a `page.evaluate`; the second is the documented failure mode and I hit it
anyway.

## A GPU distance pass is blocked by WGSL having no f64, not by the algorithm

The obvious shape of a GPU split is sound and worth writing down, because it is
the shape everyone will think of. Hamerly already narrows each point to a couple
of candidate centroids per round, so let the CPU prepare the list of distance
evaluations and batch them onto a shader. Distances are order-independent and
embarrassingly parallel, so unlike the bounds and the cluster sums they could move
without changing a bit. The sums are added in ascending point order and that is
precisely what makes the centroids stable, so they can never move.

**How much there is to move, measured.** `benches/work_split.rs` counts rather
than times, because counts are exact. On 307,200 pixels at k=32 the distance pass
is 83% to 90% of the modelled work across distinct shares from 1.3% to 64%, and
a round costs 3.9 distance evaluations per point against Lloyd's 32. So the
GPU-able share is large and the idea is not silly. The `gpu_share` column is a
model with stated per-operation weights; the counts beside it are the part to
trust.

**The blocker is the language, not the algorithm.** WGSL has no `f64`. Asking the
compiler is the only reliable way to know: `unresolved type 'f64'`, and even
`enable f16` is refused unless the adapter advertises it. So a shader can only
compute in `f32`, and the core keeps every distance in `f64`.

**And `f32` is not close enough, because the centroids are means.** All 96
components of the centroids the algorithm actually compares against are *not*
exactly representable in `f32`, because they are sums divided by counts. The
distance value differs on 100% of sampled points as a result. The chosen
centroid happened not to change on any of 20,000 sampled points, which is
reassuring and not sufficient: Hamerly builds `upper_bound` from the square root
of the distance and the convergence test compares a sum of squared movements
against a threshold, so a value that differs in the last bits can change which
branch a bound test takes and whether the run stops on round 96 or 97. **Value
equality is the invariant, not argmin equality**, and `f32` does not have it.

**An earlier version of this measurement got the wrong answer and looked
encouraging.** It compared `f32` against the centroids `kmeans_rgb` *returns*,
which are `u8`, and therefore exactly representable in `f32`, and reported zero
disagreement. The values the algorithm uses are the `f64` means on the way there.
Worth remembering as a variant of the usual trap: a golden value that is too
convenient will agree with anything.

**What it would be worth if the invariant could be relaxed.** Not nothing: with
the GPU returning only a verdict per point, an argmin and a distance, a round
costs about 1.1 MB of readback rather than the 4.4 MB of raw distances, and on
Chrome the 200 dispatches add about 6 ms against 135 to 223 ms of CPU work, so
2x to 4x is plausible. It is not measurable here for the reason above. Against
that: Firefox's measured 1038.7 μs dispatch cost puts 200 dispatches at 208 ms,
more than the entire CPU run, so it would be a browser-specific path; the values
would differ from the CPU path, so a consumer getting either one sees palette
drift; and it needs a second implementation plus a fallback. Deciding that is a
judgement about the invariant, not a measurement, and it is not mine to make
unilaterally.

## The wasm suite runs locally now

`npm run test:web` used to be CI-only, because no browser was available here. It
works now, via `scripts/run-web-tests.mjs`, which needed three things that
`wasm-pack` does not provide:

- **A browser.** `npx playwright install chromium` is the source. It is worth
  knowing that Playwright's Chromium will not start on a bare machine: it needs
  `libnspr4`, `libnss3` and `libasound2`, and `npx playwright install
  --with-deps` cannot fix that without root. The workaround is to fetch the
  Ubuntu packages with `apt-get download`, unpack them with `dpkg-deb -x` into a
  prefix, and point `BROWSER_LIBS` at it. The script puts that prefix on
  `LD_LIBRARY_PATH` for itself as well as its children, because the version
  probes launch the browser directly.
- **A chromedriver of the exact same version.** `wasm-pack` downloads one, but it
  tracks Chrome stable rather than the installed browser, so a Playwright
  Chromium 153 against its cached driver 154 fails with *cannot find Chrome
  binary*, which is a misleading way of saying the versions disagree. Chrome for
  Testing publishes an exact-version build for every Playwright release, so the
  script fetches
  `storage.googleapis.com/chrome-for-testing-public/<version>/linux64/chromedriver-linux64.zip`
  and caches it in `.browsers/`.
- **Capabilities.** chromedriver does not read the `CHROME` environment variable
  and only looks for Chrome in its own well-known locations. The test runner does
  read a `webdriver.json` from the working directory, so the script writes one
  with the binary path and the headless flags, and removes it afterwards.

**WebGPU needs a secure context, and that is the whole reason it looked
unavailable.** `navigator.gpu` is undefined on `about:blank`, which is an opaque
origin, and also on any `file:` page. Served from `http://localhost` with
`--enable-unsafe-webgpu`, `requestAdapter` and `requestDevice` both succeed in
Playwright's headless Chromium, on a software adapter. Any WebGPU work here has to
be driven through a local HTTP server; a probe that opens `about:blank` and
concludes WebGPU is unsupported is wrong.

## Testing gaps

- Native and wasm have separate `get_centroids` entropy paths, so a native test
  cannot cover the wasm one. A bug in the wasm row-slicing was introduced during
  the flat-buffer rewrite and was invisible to every native test; it was caught
  by running `docs/app.js` under a headless DOM harness. Keep that harness idea
  in mind: shim `document`, `Image` and `getContext`, stub `fetch` for `file:`
  URLs so the wasm-pack web glue can load, then import `docs/app.js` directly.
- `cargo build --target wasm32-unknown-unknown --tests` is still the only check
  that covers `tests/web.rs` at compile time, since neither `npm run check` nor
  `clippy --all-targets` builds the wasm test target. Two of its assertions were
  red in CI for exactly that reason before this was noticed.

## The playground page has a test now

`npm run test:page` runs `scripts/check-page.mjs`, which imports `docs/app.js`
under a minimal DOM and asserts on what actually got painted. It needs
`npm run pages:build` first, since it loads `docs/wasm`. It is part of
`npm run check` and runs as its own step in CI.

It exists because of a bug that reached the deployed page. The page built its
RGBA buffer with `new Uint8Array(imageData.buffer, imageData.byteOffset,
imageData.length)`. A real `ImageData` has only `data`, `width`, `height` and
`colorSpace`; `buffer`, `byteOffset` and `length` live on the
`Uint8ClampedArray` at `imageData.data`. So the buffer was **empty**,
`kmeans_rgba` returned an empty palette in about three microseconds, and all
three result canvases rendered solid black while the page still cheerfully
reported "All three results ready." The source canvas was fine, because
`drawImage` never touches that buffer.

The lesson worth keeping: a hand-written stub that is *more* capable than the
real thing hides the bug rather than catching it. The first version of this
harness gave its fake `ImageData` `buffer`, `byteOffset` and `length` getters,
which is exactly the wrong assumption, and the test passed against broken code.
The stub is now deliberately faithful, and the assertions check the painted
output rather than just the status text, because a zero-duration timing and a
black canvas are the actual symptoms.
