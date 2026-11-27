#!/usr/bin/env node

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve } from "node:path";

const PKG = resolve(import.meta.dirname, "..", "pkg");

// 1. Node ESM entry point (zero-config, synchronous top-level initSync)
const pkg = await import(new URL("../pkg/node.mjs", import.meta.url).href);

// 2. Node CJS require(esm) compatibility
const require = createRequire(import.meta.url);
const cjsPkg = require(resolve(PKG, "node.mjs"));
assert.equal(typeof cjsPkg.kmeans_rgb, "function");

// 3. Explicit ./web entry point
const webMod = await import(new URL("../pkg/kmeans_wasm.js", import.meta.url).href);
const wasmBytes = await readFile(resolve(PKG, "kmeans_wasm_bg.wasm"));
await webMod.default({ module_or_path: wasmBytes });
assert.equal(typeof webMod.kmeans_rgb, "function");

// 4. Zero-fetch ./inline entry point
const inlineMod = await import(new URL("../pkg/inline.mjs", import.meta.url).href);
await inlineMod.default();
assert.equal(typeof inlineMod.kmeans_rgb, "function");

// 5. Uint8ClampedArray support across all packed entry points
const clamped = new Uint8ClampedArray([
  255, 0, 0, 255,
  250, 5, 5, 255,
  0, 255, 0, 255,
  5, 250, 5, 255,
  0, 0, 255, 128,
  5, 5, 250, 128,
]);
const rgbaPalette = pkg.kmeans_rgba(clamped, 3, 20, 0.1, 42);
assert.equal(rgbaPalette.length, 12);
const eRgbaPalette = pkg.emeans_rgba(clamped, 3, 5, null, 42);
assert.equal(eRgbaPalette.length, 12);

const mapped = pkg.apply_palette(clamped, rgbaPalette, 4);
assert.equal(mapped.length, clamped.length);
const indices = pkg.apply_palette_indices(clamped, rgbaPalette, 4);
assert.equal(indices.length, 6);
const dithered = pkg.apply_palette_dithered(clamped, 3, 2, rgbaPalette, 4);
assert.equal(dithered.length, clamped.length);

// 6. Flat TypedArray vector APIs (Float64Array & Float32Array)
const flat64 = new Float64Array([0, 0, 0, 1, 100, 100, 100, 101]);
const res64 = pkg.kmeans_f64(flat64, 2, 2, 50, 1e-6, 7);
assert.equal(res64.k, 2);
assert.equal(res64.dimensions, 2);
assert.ok(res64.centroids instanceof Float64Array);
assert.equal(res64.centroids.length, 4);
assert.ok(res64.idxs instanceof Uint32Array);
assert.equal(res64.idxs.length, 4);

const flat32 = new Float32Array([0, 0, 0, 1, 100, 100, 100, 101]);
const res32 = pkg.emeans_f32(flat32, 2, 2, 5, null, 7);
assert.equal(res32.k, 2);
assert.equal(res32.it, 5);
assert.ok(res32.centroids instanceof Float32Array);
assert.equal(res32.centroids.length, 4);

// 7. Explicit seed determinism without mutating Math.random
const originalRandom = Math.random;
Math.random = () => {
  throw new Error("Math.random must not be called when an explicit seed is supplied");
};
try {
  const sample = new Uint8Array(300 * 3);
  for (let i = 0; i < 300; i += 1) {
    sample[i * 3] = (i * 37) & 0xff;
    sample[i * 3 + 1] = (i * 73) & 0xff;
    sample[i * 3 + 2] = (i * 19) & 0xff;
  }
  const p1 = pkg.kmeans_rgb(sample, 8, 30, 0.1, 99);
  const p2 = pkg.kmeans_rgb(sample, 8, 30, 0.1, 99);
  assert.deepEqual(p1, p2);

  const e1 = pkg.emeans_rgb(sample, 8, 4, null, 99);
  const e2 = pkg.emeans_rgb(sample, 8, 4, null, 99);
  assert.deepEqual(e1, e2);
} finally {
  Math.random = originalRandom;
}

// 8. CSP safety: result.test works without dynamic `new Function`
const originalFunction = globalThis.Function;
globalThis.Function = function BlockedByCsp() {
  throw new EvalError("EvalError: Refused to evaluate a string as JavaScript (CSP)");
};
try {
  const vecRes = pkg.kmeans([[0, 0], [0, 1], [50, 50], [50, 51]], 2, 20, 0.001, 1);
  assert.notEqual(vecRes.test([0.2, 0.2]), vecRes.test([49.8, 49.8]));
  const eVecRes = pkg.emeans([[0, 0], [0, 1], [50, 50], [50, 51]], 2, 3, 2, 1);
  assert.notEqual(eVecRes.test([0.2, 0.2]), eVecRes.test([49.8, 49.8]));
} finally {
  globalThis.Function = originalFunction;
}

// 9. Error instances and validation
assert.throws(
  () => pkg.kmeans_rgb(new Uint8Array(6), 2, 10, Number.NaN),
  (err) => err instanceof Error && err instanceof RangeError,
);
assert.throws(
  () => pkg.kmeans([[1, 2], [3, Number.NaN]], 2, 10),
  (err) => err instanceof Error && err instanceof RangeError,
);
assert.throws(
  () => pkg.kmeans([123, [3, 4]], 2, 10),
  (err) => err instanceof Error && err instanceof TypeError,
);
assert.throws(
  () => pkg.kmeans_f64(new Float64Array([1, 2, 3]), 2, 2, 10),
  (err) => err instanceof Error && err instanceof RangeError,
);

console.log("package smoke tests passed");
