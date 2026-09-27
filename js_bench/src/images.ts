import { apply_palette, kmeans_rgba, kmeans_rgb } from "kmeans-wasm";
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
 * skmeans gets its own, much smaller repeat count.
 *
 * On a megapixel it takes 2 seconds at k=8 and 12 at k=32, so the usual eight
 * measured runs after two warm-ups would be twenty minutes per image. Two runs is
 * enough to have a warm-up and a sample, and the sample is seconds long rather
 * than milliseconds, so the relative noise is much lower than the absolute
 * numbers suggest. The counts are reported with the table rather than hidden.
 */
const SKMEANS_REPEATS = 1;
const SKMEANS_WARMUPS = 1;

const IMAGE_DIRECTORY = join(__dirname, "..", "images");

interface Loaded {
  name: string;
  pixels: number;
  rgb: Uint8Array;
  rgba: Uint8Array;
  points: number[][];
  distinct: number;
}

interface Row {
  image: string;
  pixels: number;
  distinctShare: number;
  colors: number;
  rgbMs: number;
  rgbaMs: number;
  skmeansMs: number;
  /** `apply_palette`, mapping the palette back onto the image. */
  mapMs: number;
  /** The JavaScript loop `apply_palette` replaces, with the same Map cache the playground used. */
  jsMapMs: number;
}

/**
 * The mapping half of quantization, written the way a caller had to before
 * `apply_palette`: a per-pixel search over the palette, memoised in a Map keyed
 * on the packed colour. This is the honest baseline, because it is the code the
 * README used to ask people to write and what `docs/app.js` ran before the entry
 * point existed.
 *
 * Returns RGBA, like `apply_palette`, so the two can be compared byte for byte
 * rather than only by timing.
 */
function jsMapping(pixels: Uint8Array, pixelsCount: number, palette: Uint8Array): Uint8Array {
  const output = new Uint8Array(pixelsCount * 4);
  const cache = new Map<number, number>();
  for (let pixel = 0; pixel < pixelsCount; pixel += 1) {
    const offset = pixel * 3;
    const out = pixel * 4;
    const key = (pixels[offset] << 16) | (pixels[offset + 1] << 8) | pixels[offset + 2];
    let paletteOffset = cache.get(key);
    if (paletteOffset === undefined) {
      let best = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let entry = 0; entry < palette.length; entry += 3) {
        const dr = pixels[offset] - palette[entry];
        const dg = pixels[offset + 1] - palette[entry + 1];
        const db = pixels[offset + 2] - palette[entry + 2];
        const distance = dr * dr + dg * dg + db * db;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = entry;
        }
      }
      paletteOffset = best;
      cache.set(key, paletteOffset);
    }
    output[out] = palette[paletteOffset];
    output[out + 1] = palette[paletteOffset + 1];
    output[out + 2] = palette[paletteOffset + 2];
    output[out + 3] = 255;
  }
  return output;
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function measureWith(warmups: number, repeats: number, operation: () => unknown): number {
  for (let index = 0; index < warmups; index += 1) {
    operation();
  }
  const samples: number[] = [];
  for (let index = 0; index < repeats; index += 1) {
    const start = performance.now();
    operation();
    samples.push(performance.now() - start);
  }
  return median(samples);
}

function measure(operation: () => unknown): number {
  return measureWith(WARMUPS, REPEATS, operation);
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

  return { name, pixels, rgb, rgba, points, distinct: seen.size };
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
    // Kept so the mapping can use the palette the clustering actually produced.
    const palette = kmeans_rgb(image.rgb, colors, MAX_ITERATIONS, CONVERGENCE_THRESHOLD);
    const rgbMs = measure(() =>
      kmeans_rgb(image.rgb, colors, MAX_ITERATIONS, CONVERGENCE_THRESHOLD),
    );
    const rgbaMs = measure(() =>
      kmeans_rgba(image.rgba, colors, MAX_ITERATIONS, CONVERGENCE_THRESHOLD),
    );
    const skmeansMs = measureWith(
      SKMEANS_WARMUPS,
      SKMEANS_REPEATS,
      () => skmeans(image.points, colors, RANDOM_SEEDING, MAX_ITERATIONS),
    );
    // The mapping is measured against the loop it replaces, and the two are
    // compared byte for byte, so this doubles as a correctness check on every
    // image and every colour count rather than only a timing.
    const fromWasm = apply_palette(image.rgb, palette, 3);
    const fromJavaScript = jsMapping(image.rgb, image.pixels, palette);
    if (fromWasm.length !== fromJavaScript.length) {
      throw new Error(`${image.name} at k=${colors}: lengths differ`);
    }
    for (let index = 0; index < fromWasm.length; index += 1) {
      if (fromWasm[index] !== fromJavaScript[index]) {
        throw new Error(
          `${image.name} at k=${colors}: apply_palette differs from the JavaScript loop at byte ${index}`,
        );
      }
    }

    const mapMs = measure(() => apply_palette(image.rgb, palette, 3));
    const jsMapMs = measure(() => jsMapping(image.rgb, image.pixels, palette));

    rows.push({
      image: image.name.replace(/\.jpg$/, ""),
      pixels: image.pixels,
      distinctShare: (100 * image.distinct) / image.pixels,
      colors,
      rgbMs,
      rgbaMs,
      skmeansMs,
      mapMs,
      jsMapMs,
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
    "skmeans (ms)": Number(row.skmeansMs.toFixed(2)),
    "apply_palette (ms)": Number(row.mapMs.toFixed(2)),
    "js loop (ms)": Number(row.jsMapMs.toFixed(2)),
    "map speed-up": `${(row.jsMapMs / row.mapMs).toFixed(2)}x`,
  })),
);

console.log("\nMarkdown table");
console.log(
  "| Image | Pixels | Distinct | Colors | kmeans_rgb | kmeans_rgba | skmeans | Speed-up |",
);
console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const row of rows) {
  console.log(
    `| ${row.image} | ${row.pixels.toLocaleString("en-US")} | ${row.distinctShare.toFixed(1)}% | ${row.colors} | ${row.rgbMs.toFixed(2)} ms | ${row.rgbaMs.toFixed(2)} ms | ${row.skmeansMs.toFixed(2)} ms | ${(row.skmeansMs / row.rgbMs).toFixed(1)}x |`,
  );
}
console.log(
  `\nMedian of ${REPEATS} runs after ${WARMUPS} warm-ups; max iterations: ${MAX_ITERATIONS}; convergence threshold: ${CONVERGENCE_THRESHOLD}, because skmeans has no threshold parameter.`,
);
console.log(
  `Every column is a full-image measurement of the same pixels. kmeans-wasm is the median of ${REPEATS} runs after ${WARMUPS} warm-ups; skmeans is the median of ${SKMEANS_REPEATS} after ${SKMEANS_WARMUPS}, because it costs seconds per run on a megapixel.`,
);

console.log("\nMapping table (apply_palette against the JavaScript loop it replaces)");
console.log("| Image | Pixels | Distinct | Colors | apply_palette | js loop | Speed-up |");
console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const row of rows) {
  console.log(
    `| ${row.image} | ${row.pixels.toLocaleString("en-US")} | ${row.distinctShare.toFixed(1)}% | ${row.colors} | ${row.mapMs.toFixed(2)} ms | ${row.jsMapMs.toFixed(2)} ms | ${(row.jsMapMs / row.mapMs).toFixed(2)}x |`,
  );
}
console.log(
  "Both columns were checked to be byte-identical on every row, so the speed-up is on identical output rather than on two things that merely look similar.",
);
