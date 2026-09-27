#!/usr/bin/env node

// Can a WebGPU compute kernel map a palette onto an image, and is it worth it?
//
// Two separate questions, and the second one is the one that decides it.
//
// 1. Is it *correct*? A kernel has to be byte-identical to `apply_palette`,
//    because every consumer here depends on a reproducible image and not just a
//    reproducible palette. That is testable anywhere.
//
// 2. Is it *worth it*? Mapping is the second half of quantization, and it is
//    worth a GPU only in proportion to how much of the pipeline it is. That is
//    also testable anywhere.
//
// What this cannot do is measure speed. WebGPU needs a secure context, so it only
// appears on http://localhost and never on about:blank or a file: URL. And on a
// machine with no hardware GPU the only adapter available is SwiftShader, which
// is a software rasteriser: 4.8 ms for a million trivial invocations, against
// roughly 15 microseconds for real hardware. Any timing printed here therefore
// describes a CPU pretending to be a GPU, and is a floor rather than a
// prediction.
//
// Run with: node scripts/probe-gpu-mapping.mjs
// Needs: npx playwright install chromium, and BROWSER_LIBS set if the browser's
// system libraries are not installed system-wide. See scripts/run-web-tests.mjs.

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readdirSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const K = 32;
const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".jpg": "image/jpeg",
};

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
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log(`  [page error] ${e.message}`));
await page.goto(`${base}/docs/index.html`);

// One u32 per pixel for the shader, and one tightly packed RGB byte stream for
// the wasm entry points. These are not interchangeable: a u32-per-pixel buffer
// read as 3-component bytes puts every pixel three bytes out of step, which
// looks exactly like a GPU correctness bug and is not one.
const SHADER = /* wgsl */ `
struct Params { count: u32, k: u32, components: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       pixels:  array<u32>;
@group(0) @binding(1) var<storage, read>       palette: array<u32>;
@group(0) @binding(2) var<storage, read_write> out:     array<u32>;
@group(0) @binding(3) var<uniform>             params:  Params;

fn distance(p: u32, q: u32, shift: u32) -> u32 {
  let d0 = i32((p >> shift) & 0xFFu) - i32((q >> shift) & 0xFFu);
  let d1 = i32((p >> (shift - 8u)) & 0xFFu) - i32((q >> (shift - 8u)) & 0xFFu);
  let d2 = i32(p & 0xFFu) - i32(q & 0xFFu);
  return u32(d0 * d0 + d1 * d1 + d2 * d2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.count) { return; }
  let p = pixels[i];
  let shift = (params.components - 1u) * 8u;

  // Ascending entry with a strict comparison, which is what puts a tie on the
  // lower one. That, and integer arithmetic, is the whole reason this can be
  // byte-identical to the CPU at all.
  var best = 0u;
  var bestDistance = 0xFFFFFFFFu;
  for (var e = 0u; e < params.k; e = e + 1u) {
    let d = distance(p, palette[e], shift);
    if (d < bestDistance) { bestDistance = d; best = e; }
  }
  out[i] = palette[best];
}`;

const adapter = await page.evaluate(async () => {
  if (typeof navigator.gpu === "undefined") return { secure: window.isSecureContext, gpu: false };
  const found = await navigator.gpu.requestAdapter();
  if (!found) return { secure: window.isSecureContext, gpu: true, adapter: false };
  return {
    secure: window.isSecureContext,
    gpu: true,
    adapter: true,
    // Empty in this build, so there is no vendor string to identify the adapter
    // with. The timing below is the only evidence about what it is.
    info: { ...(found.info ?? {}) },
  };
});
console.log(`adapter: ${JSON.stringify(adapter)}\n`);
if (!adapter.adapter) {
  console.error("no WebGPU adapter; nothing to measure");
  await browser.close();
  server.close();
  process.exit(1);
}

const floor = await page.evaluate(async () => {
  const device = await (await navigator.gpu.requestAdapter()).requestDevice();
  const module = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> d: array<u32>;
           @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
             d[i.x] = i.x * 2u;
           }`,
  });
  const n = 1 << 20;
  const buffer = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
  const dispatch = () => {
    const e = device.createCommandEncoder();
    const p = e.beginComputePass();
    p.setPipeline(pipeline);
    p.setBindGroup(0, bind);
    p.dispatchWorkgroups(n / 64);
    p.end();
    device.queue.submit([e.finish()]);
  };
  for (let i = 0; i < 3; i += 1) dispatch();
  await device.queue.onSubmittedWorkDone();
  const samples = [];
  for (let i = 0; i < 9; i += 1) {
    const t = performance.now();
    dispatch();
    await device.queue.onSubmittedWorkDone();
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  return samples[0];
});
console.log(`one million trivial invocations: ${floor.toFixed(2)} ms`);
console.log(
  "  Real hardware is roughly 15 microseconds for the same work, so anything an order of\n" +
    "  magnitude or two above that is a software adapter and the GPU column below is a floor.\n",
);

const images = readdirSync(resolve(ROOT, "js_bench/images")).filter((f) => f.endsWith(".jpg")).sort();
console.log(`image            pixels  distinct   k  identical  cluster_ms   map_ms   gpu_ms  map_share`);
console.log(`${"-".repeat(85)}`);

const shares = [];

for (const file of images) {
  const result = await page.evaluate(
    async ({ url, shader, k }) => {
      const module = await import("/docs/wasm/kmeans_wasm.js");
      await module.default();
      const { kmeans_rgb, apply_palette } = module;

      // kmeans_rgb reads Math.random, so an unpinned run compares two different
      // initialisations and the clustering time moves with the draw. The page
      // pins it the same way before every measurement.
      let state = 0x9e3779b9 >>> 0;
      Math.random = () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };

      const bitmap = await createImageBitmap(await (await fetch(url)).blob());
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const context = canvas.getContext("2d");
      context.drawImage(bitmap, 0, 0);
      const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
      const count = bitmap.width * bitmap.height;

      const rgb = new Uint8Array(count * 3);
      const packed = new Uint32Array(count);
      for (let i = 0; i < count; i += 1) {
        const o = i * 4;
        rgb[i * 3] = data[o];
        rgb[i * 3 + 1] = data[o + 1];
        rgb[i * 3 + 2] = data[o + 2];
        packed[i] = ((data[o] << 16) | (data[o + 1] << 8) | data[o + 2]) >>> 0;
      }
      const distinct = new Set(packed).size;

      // Clustering is timed because mapping is only worth accelerating in
      // proportion to how much of the pipeline it is.
      const clusterSamples = [];
      for (let run = 0; run < 5; run += 1) {
        const t0 = performance.now();
        kmeans_rgb(rgb, k, 100, 0.1);
        clusterSamples.push(performance.now() - t0);
      }
      clusterSamples.sort((a, b) => a - b);

      const paletteBytes = kmeans_rgb(rgb, k, 100, 0.1);
      const palette = new Uint32Array(k);
      for (let e = 0; e < k; e += 1) {
        const o = e * 3;
        palette[e] = ((paletteBytes[o] << 16) | (paletteBytes[o + 1] << 8) | paletteBytes[o + 2]) >>> 0;
      }

      const rgba = apply_palette(rgb, paletteBytes, 3);
      const cpu = new Uint32Array(count);
      for (let i = 0; i < count; i += 1) {
        const o = i * 4;
        cpu[i] = ((rgba[o] << 16) | (rgba[o + 1] << 8) | rgba[o + 2]) >>> 0;
      }

      const device = await (await navigator.gpu.requestAdapter()).requestDevice();
      const shaderModule = device.createShaderModule({ code: shader });
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "main" } });

      // Every buffer written here needs COPY_DST. A buffer created as UNIFORM
      // alone silently fails the write, the count stays zero, the kernel's guard
      // returns before writing anything, and the readback is all zeros. That is
      // the exact shape this failure takes, and it looks like a dead kernel.
      const makeBuffer = (data, usage) => {
        const buffer = device.createBuffer({
          size: Math.max(16, data.byteLength),
          usage: usage | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      const pixelsBuffer = makeBuffer(packed, GPUBufferUsage.STORAGE);
      const paletteBuffer = makeBuffer(palette, GPUBufferUsage.STORAGE);
      const paramsBuffer = makeBuffer(new Uint32Array([count, k, 3, 0]), GPUBufferUsage.UNIFORM);
      const outBuffer = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: pixelsBuffer } },
          { binding: 1, resource: { buffer: paletteBuffer } },
          { binding: 2, resource: { buffer: outBuffer } },
          { binding: 3, resource: { buffer: paramsBuffer } },
        ],
      });

      const dispatch = () => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(count / 64));
        pass.end();
        encoder.copyBufferToBuffer(outBuffer, 0, readback, 0, count * 4);
        device.queue.submit([encoder.finish()]);
      };

      dispatch();
      await readback.mapAsync(GPUMapMode.READ);
      const gpu = new Uint32Array(readback.getMappedRange().slice(0));
      readback.unmap();

      let mismatches = 0;
      for (let i = 0; i < count; i += 1) if (gpu[i] !== cpu[i]) mismatches += 1;

      const time = async (fn, runs) => {
        for (let i = 0; i < 2; i += 1) await fn();
        const samples = [];
        for (let i = 0; i < runs; i += 1) {
          const t = performance.now();
          await fn();
          samples.push(performance.now() - t);
        }
        samples.sort((a, b) => a - b);
        return samples[0];
      };
      const gpuMs = await time(async () => {
        dispatch();
        await device.queue.onSubmittedWorkDone();
      }, 5);
      const mapMs = await time(async () => apply_palette(rgb, paletteBytes, 3), 5);

      return {
        count,
        distinct,
        mismatches,
        clusterMs: clusterSamples[0],
        mapMs,
        gpuMs,
        share: (100 * mapMs) / (clusterSamples[0] + mapMs),
      };
    },
    { url: `${base}/js_bench/images/${file}`, shader: SHADER, k: K },
  );

  shares.push(result.share);
  console.log(
    `${file.replace(".jpg", "").padEnd(13)} ${String(result.count).padStart(9)} ${String(result.distinct).padStart(9)} ${String(K).padStart(4)} ` +
      `${(result.mismatches === 0 ? "yes" : `NO (${result.mismatches})`).padStart(9)} ` +
      `${result.clusterMs.toFixed(1).padStart(10)} ${result.mapMs.toFixed(1).padStart(8)} ${result.gpuMs.toFixed(1).padStart(8)} ` +
      `${result.share.toFixed(0).padStart(8)}%`,
  );
}

const low = Math.min(...shares).toFixed(0);
const high = Math.max(...shares).toFixed(0);
console.log(
  `\nThe last column is the ceiling on what a GPU mapping path could win: mapping is\n` +
    `${low}% to ${high}% of the pipeline and clustering is the rest. Making mapping free\n` +
    `would be a ${(100 / (100 - low)).toFixed(2)}x to ${(100 / (100 - high)).toFixed(2)}x end-to-end win, on hardware this\n` +
    "environment cannot measure.",
);

await browser.close();
server.close();
