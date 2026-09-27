import { kmeans_rgba, kmeans_rgb } from "kmeans-wasm";
import jpeg from "jpeg-js";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
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
 * still, but skmeans cannot be given the same setting so it is not measured here.
 */
const MAX_ITERATIONS = 100;
const CONVERGENCE_THRESHOLD = 0;

/** skmeans' own random seeding, named explicitly so both sides match. */
const RANDOM_SEEDING = "kmrand";

const REPEATS = 8;
const WARMUPS = 2;
const CLUSTER_COUNTS = [8, 16, 32, 64];

/**
 * skmeans is a JavaScript implementation taking `number[][]`, so it cannot be
 * run over a whole photograph in reasonable time: at 922,560 points and 64
 * clusters over 100 iterations that is billions of interpreted distance
 * evaluations. It is timed over a fixed evenly spaced subsample instead, which is
 * reported in its own column and is not comparable to the full-image columns.
 */
const SKMEANS_SUBSAMPLE = 8_000;

const IMAGE_DIRECTORY = join(__dirname, "..", "images");

interface Loaded {
  name: string;
  pixels: number;
  rgb: Uint8Array;
  rgba: Uint8Array;
  points: number[][];
  distinct: number;
  subsample: number[][];
}

interface Row {
  image: string;
  pixels: number;
  distinctShare: number;
  colors: number;
  rgbMs: number;
  rgbaMs: number;
  skmeansSubsampleMs: number;
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

function load(name: string): Loaded {
  const decoded = jpeg.decode(readFileSync(join(IMAGE_DIRECTORY, name)), {
    useTArray: true,
  });
  const pixels = decoded.width * decoded.height;

  const rgb = new Uint8Array(pixels * 3);
  const rgba = new Uint8Array(pixels * 4);
  const points: number[][] = new Array(pixels);
  const seen = new Set<number>();

  for (let index = 0; index < pixels; index += 1) {
    const from = index * 4;
    const to = index * 3;
    const r = decoded.data[from];
    const g = decoded.data[from + 1];
    const b = decoded.data[from + 2];
    rgb[to] = r;
    rgb[to + 1] = g;
    rgb[to + 2] = b;
    rgba[from] = r;
    rgba[from + 1] = g;
    rgba[from + 2] = b;
    rgba[from + 3] = decoded.data[from + 3];
    points[index] = [r, g, b];
    seen.add((r << 16) | (g << 8) | b);
  }

  const stride = Math.max(1, Math.floor(pixels / SKMEANS_SUBSAMPLE));
  const subsample: number[][] = [];
  for (let index = 0; index < pixels; index += stride) {
    subsample.push(points[index]);
  }

  return { name, pixels, rgb, rgba, points, distinct: seen.size, subsample };
}

const images = readdirSync(IMAGE_DIRECTORY)
  .filter((name) => name.endsWith(".jpg"))
  .sort()
  .map((name) => load(name));

console.log("\nReal image benchmark");
console.table(
  images.map((image) => ({
    image: image.name.replace(/\.jpg$/, ""),
    size: `${image.pixels.toLocaleString("en-US")} px`,
    distinct: image.distinct.toLocaleString("en-US"),
    "distinct share": `${((100 * image.distinct) / image.pixels).toFixed(1)}%`,
    "point-set shrink": `${(image.pixels / image.distinct).toFixed(1)}x`,
  })),
);

const rows: Row[] = [];
for (const image of images) {
  for (const colors of CLUSTER_COUNTS) {
    const rgbMs = measure(() =>
      kmeans_rgb(image.rgb, colors, MAX_ITERATIONS, CONVERGENCE_THRESHOLD),
    );
    const rgbaMs = measure(() =>
      kmeans_rgba(image.rgba, colors, MAX_ITERATIONS, CONVERGENCE_THRESHOLD),
    );
    const skmeansSubsampleMs = measure(() =>
      skmeans(image.subsample, colors, RANDOM_SEEDING, MAX_ITERATIONS),
    );
    rows.push({
      image: image.name.replace(/\.jpg$/, ""),
      pixels: image.pixels,
      distinctShare: (100 * image.distinct) / image.pixels,
      colors,
      rgbMs,
      rgbaMs,
      skmeansSubsampleMs,
    });
  }
}

console.log("\nTimings");
console.table(
  rows.map((row) => ({
    image: row.image,
    pixels: row.pixels.toLocaleString("en-US"),
    distinct: `${row.distinctShare.toFixed(1)}%`,
    colors: row.colors,
    "kmeans_rgb (ms)": Number(row.rgbMs.toFixed(2)),
    "kmeans_rgba (ms)": Number(row.rgbaMs.toFixed(2)),
    "skmeans 8k (ms)": Number(row.skmeansSubsampleMs.toFixed(2)),
  })),
);

console.log("\nMarkdown table");
console.log(
  "| Image | Pixels | Distinct | Colors | kmeans_rgb | kmeans_rgba | skmeans (8k subsample) |",
);
console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const row of rows) {
  console.log(
    `| ${row.image} | ${row.pixels.toLocaleString("en-US")} | ${row.distinctShare.toFixed(1)}% | ${row.colors} | ${row.rgbMs.toFixed(2)} ms | ${row.rgbaMs.toFixed(2)} ms | ${row.skmeansSubsampleMs.toFixed(2)} ms |`,
  );
}
console.log(
  `\nMedian of ${REPEATS} runs after ${WARMUPS} warm-ups; max iterations: ${MAX_ITERATIONS}; convergence threshold: ${CONVERGENCE_THRESHOLD}, because skmeans has no threshold parameter.`,
);
console.log(
  "The skmeans column is a subsample and is not comparable to the full-image columns.",
);
