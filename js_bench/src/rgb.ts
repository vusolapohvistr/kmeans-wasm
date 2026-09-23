import { kmeans_rgb } from "kmeans-wasm";
import skmeans from "skmeans";

const MAX_ITERATIONS = 100;
const CONVERGENCE_THRESHOLD = 0.1;
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

function makeRgbData(pixelCount: number): RgbData {
  let state = 0x12345678;
  const rgb = new Uint8Array(pixelCount * 3);
  const points = new Array<number[]>(pixelCount);

  const nextByte = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state & 0xff;
  };

  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 3;
    const point = [nextByte(), nextByte(), nextByte()];
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
    skmeans(data.points, colors, undefined, MAX_ITERATIONS),
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
