import { kmeans_rgb } from "kmeans-wasm";
import skmeans from "skmeans";

/**
 * skmeans has no convergence threshold. Its signature is
 * `(data, k, centroids, iterations, distance)` and it runs until no point
 * changes cluster or the iteration cap is hit.
 *
 * So the only stopping rule both libraries can share is no early exit at all,
 * and that is what the comparison uses. It is also the conservative direction:
 * kmeans-wasm then does strictly more work than skmeans, which stops as soon as
 * the assignments settle, so the speed-ups below are a floor rather than a best
 * case. The packed entry points default to 0.1 in normal use, which is faster
 * still, but skmeans cannot be given the same setting so it is not measured
 * here.
 */
const MAX_ITERATIONS = 100;
const CONVERGENCE_THRESHOLD = 0;

/** skmeans' own random seeding, named explicitly so both sides match. */
const RANDOM_SEEDING = "kmrand";

const REPEATS = 8;
const WARMUPS = 2;

interface RgbData {
  rgb: Uint8Array;
  points: number[][];
}

interface RgbCase {
  pixels: number;
  colors: number;
}

interface BenchmarkRow extends RgbCase {
  wasmMs: number;
  skmeansMs: number;
  speedup: number;
}

/**
 * A deterministic generator that actually produces distinct values.
 *
 * The obvious choice, the low byte of a 32-bit LCG, is a trap: an LCG's low bits
 * have a period of 256, so it emits the same 256 byte values forever. That made
 * this harness measure a 256-colour repeating pattern while calling it random
 * pixels, which is invisible until something starts exploiting repetition and the
 * numbers become implausible.
 *
 * mulberry32 is an LCG too, but the mixing step before each output means its low
 * bits are well behaved. Taking the top 24 bits of each draw gives a full byte
 * triple per draw, so 3-component points are distinct with overwhelming
 * probability.
 */
function makeGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    mixed = (mixed ^ (mixed >>> 14)) >>> 0;
    return (mixed >>> 8) & 0xffffff;
  };
}

function makeRgbData(pixelCount: number): RgbData {
  const rgb = new Uint8Array(pixelCount * 3);
  const points = new Array<number[]>(pixelCount);
  const next = makeGenerator(0x12345678);

  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 3;
    const point = [next() & 0xff, (next() >>> 8) & 0xff, (next() >>> 16) & 0xff];
    rgb.set(point, offset);
    points[index] = point;
  }

  return { rgb, points };
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function measure(operation: () => unknown): number {
  for (let index = 0; index < WARMUPS; index += 1) {
    operation();
  }

  const samples: number[] = [];
  for (let index = 0; index < REPEATS; index += 1) {
    const start = performance.now();
    operation();
    samples.push(performance.now() - start);
  }

  return median(samples);
}

function runCase({ pixels, colors }: RgbCase): BenchmarkRow {
  const data = makeRgbData(pixels);
  // Exercise the dedicated RGB path rather than the general Array-based API.
  const wasmMs = measure(() =>
    kmeans_rgb(data.rgb, colors, MAX_ITERATIONS, CONVERGENCE_THRESHOLD),
  );
  const skmeansMs = measure(() =>
    skmeans(data.points, colors, RANDOM_SEEDING, MAX_ITERATIONS),
  );

  return {
    pixels,
    colors,
    wasmMs,
    skmeansMs,
    speedup: skmeansMs / wasmMs,
  };
}

const cases: RgbCase[] = [
  { pixels: 1_000, colors: 2 },
  { pixels: 10_000, colors: 4 },
  { pixels: 10_000, colors: 16 },
  { pixels: 100_000, colors: 2 },
  { pixels: 100_000, colors: 8 },
  { pixels: 100_000, colors: 32 },
];

const rows = cases.map(runCase);

console.log("\nRGB benchmark");
console.table(
  rows.map(({ pixels, colors, wasmMs, skmeansMs, speedup }) => ({
    pixels: pixels.toLocaleString("en-US"),
    colors,
    "kmeans_rgb (ms)": Number(wasmMs.toFixed(2)),
    "skmeans (ms)": Number(skmeansMs.toFixed(2)),
    "speed-up": `${speedup.toFixed(1)}x`,
  })),
);

console.log("\nMarkdown table");
console.log("| Test | Pixels | Colors | kmeans_rgb | skmeans | Speed-up |");
console.log("| --- | ---: | ---: | ---: | ---: | ---: |");
for (const row of rows) {
  console.log(
    `| RGB random pixels | ${row.pixels.toLocaleString("en-US")} | ${row.colors} | ${row.wasmMs.toFixed(2)} ms | ${row.skmeansMs.toFixed(2)} ms | ${row.speedup.toFixed(1)}x |`,
  );
}
console.log(`\nMedian of ${REPEATS} runs after ${WARMUPS} warm-ups; max iterations: ${MAX_ITERATIONS}.`);
