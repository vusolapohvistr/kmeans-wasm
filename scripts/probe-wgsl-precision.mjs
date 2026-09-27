// Can a WGSL kernel compute the distance pass bit-identically to the CPU?
//
// The clustering core keeps every distance in f64. So the question is whether
// WGSL has f64 at all, and if it does not, how far f32 lands from f64 in
// practice: a different argmin in a near-tie is a different centroid, and this
// crate's central invariant is that the output is bit-for-bit stable.
//
// Run with: node scripts/probe-wgsl-precision.mjs

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".jpg": "image/jpeg" };
const server = createServer(async (req, res) => {
  const file = resolve(ROOT, "." + decodeURIComponent(req.url.split("?")[0]));
  let body;
  try {
    body = await readFile(file);
  } catch {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream" });
  res.end(body);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));

const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage();
page.on("console", (m) => console.log(`  [page] ${m.text()}`));
await page.goto(`http://127.0.0.1:${server.address().port}/docs/index.html`);

const report = await page.evaluate(async () => {
  const device = await (await navigator.gpu.requestAdapter()).requestDevice();
  const out = {};

  // Does WGSL have f64? Ask the compiler rather than the documentation.
  const tryCompile = async (label, code) => {
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    return { label, compiles: errors.length === 0, firstError: errors[0]?.message };
  };

  out.precision = [];
  out.precision.push(
    await tryCompile(
      "f64 in a compute kernel",
      `@group(0) @binding(0) var<storage, read_write> d: array<f64>;
       @compute @workgroup_size(1) fn main() { d[0] = 1.0; }`,
    ),
  );
  out.precision.push(
    await tryCompile(
      "f16 in a compute kernel",
      `enable f16;
       @group(0) @binding(0) var<storage, read_write> d: array<f16>;
       @compute @workgroup_size(1) fn main() { d[0] = f16(1.0); }`,
    ),
  );
  out.precision.push(
    await tryCompile(
      "f32 for reference",
      `@group(0) @binding(0) var<storage, read_write> d: array<f32>;
       @compute @workgroup_size(1) fn main() { d[0] = 1.0; }`,
    ),
  );

  // If f64 is out, how wrong is f32? The centroids the algorithm actually
  // compares against are f64 *means*, not the u8 values kmeans_rgb finally
  // returns, and a mean of u8 values is generally not representable in f32. So
  // the question is whether that inexactness ever changes which centroid wins,
  // and whether it changes the distance values that the bounds are built from.
  const wasm = await import("/docs/wasm/kmeans_wasm.js");
  await wasm.default();
  const { kmeans } = wasm;

  const bitmap = await createImageBitmap(
    await (await fetch("/js_bench/images/blue-marble.jpg")).blob(),
  );
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  const total = bitmap.width * bitmap.height;

  const k = 32;
  let state = 0x9e3779b9 >>> 0;
  Math.random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  // A subsample, because the general entry point takes JavaScript arrays and the
  // rate being measured is a per-point property. Reported with the sample size
  // rather than as if it were the whole image.
  const SAMPLE = 20_000;
  const stride = Math.max(1, Math.floor(total / SAMPLE));
  const points = [];
  for (let i = 0; i < total && points.length < SAMPLE; i += stride) {
    const o = i * 4;
    points.push([data[o], data[o + 1], data[o + 2]]);
  }
  const result = kmeans(points, k, 30, 0);
  const cent = result.centroids.map((c) => [c[0], c[1], c[2]]);
  out.roundsRun = result.it;
  out.sampleSize = points.length;

  // Are the centroids the algorithm actually used expressible in f32 at all?
  let inexpressible = 0;
  for (const c of cent) for (const v of c) if (Math.fround(v) !== v) inexpressible += 1;
  out.centroidComponents = cent.length * 3;
  out.centroidComponentsNotF32Exact = inexpressible;

  const f32c = cent.map(([r, g, b]) => [Math.fround(r), Math.fround(g), Math.fround(b)]);
  const fr = Math.fround;
  let differentArgmin = 0;
  let differentValue = 0;
  let largestRelativeGap = 0;
  for (const x of points) {
    let best64 = 0;
    let d64 = Infinity;
    for (let e = 0; e < k; e += 1) {
      let d = 0;
      for (let c = 0; c < 3; c += 1) {
        const diff = x[c] - cent[e][c];
        d += diff * diff;
      }
      if (d < d64) { d64 = d; best64 = e; }
    }
    let best32 = 0;
    let d32 = Infinity;
    for (let e = 0; e < k; e += 1) {
      let d = 0;
      for (let c = 0; c < 3; c += 1) {
        const diff = fr(fr(x[c]) - f32c[e][c]);
        d = fr(d + fr(diff * diff));
      }
      if (d < d32) { d32 = d; best32 = e; }
    }
    if (d32 !== d64) differentValue += 1;
    if (best32 !== best64) differentArgmin += 1;
    // How close the nearest call is, as a fraction of the distance. A small
    // margin is what makes an argmin flip likely, so it is worth reporting.
    if (d64 > 0) {
      let second = Infinity;
      for (let e = 0; e < k; e += 1) {
        if (e === best64) continue;
        let d = 0;
        for (let c = 0; c < 3; c += 1) {
          const diff = x[c] - cent[e][c];
          d += diff * diff;
        }
        if (d < second) second = d;
      }
      const gap = (second - d64) / second;
      if (gap < largestRelativeGap) largestRelativeGap = gap;
    }
  }
  out.pixelsExamined = points.length;
  out.pixelsWhereF32ValueDiffers = differentValue;
  out.pixelsWhereF32PicksAnotherCentroid = differentArgmin;
  out.f32ValueDisagreementRate = `${((100 * differentValue) / points.length).toFixed(2)}%`;
  out.f32ArgminDisagreementRate = `${((100 * differentArgmin) / points.length).toFixed(2)}%`;
  out.closestRelativeMarginSeen = Number(largestRelativeGap.toExponential(2));

  return out;
});

console.log(JSON.stringify(report, null, 2));
await browser.close();
server.close();
