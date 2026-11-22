import { emeans_rgb, emeans_rgba, kmeans_rgb } from "kmeans-wasm";
import jpeg from "jpeg-js";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

/**
 * e-means against Lloyd on the four real photographs.
 *
 * The comparison that matters is quality at equal work, and the paper's claim is
 * that a handful of passes matches a converged Lloyd run. So both sides are
 * seeded from the same stream, which makes their initial centroids identical,
 * and the table reports weighted MSE over the image's distinct colours next to
 * the wall time.
 *
 * Lloyd runs the shipped packed entry point to its 100-iteration cap with no
 * early convergence, matching the other reference tables. e-means has no
 * threshold; it stops after the requested passes.
 */
const MAX_ITERATIONS = 100;
const CONVERGENCE_THRESHOLD = 0;
const EPOCH_VARIANTS = [3, 5, 10] as const;
const REPORTED_EPOCHS = 5;

const REPEATS = 8;
const WARMUPS = 2;

/**
 * One fixed stream for every measured call. Both entry points read
 * `Math.random`, so pinning it before the call makes the initial draw identical
 * and every repeat comparable.
 */
const SEED = 0x1234abcd;

const CLUSTER_COUNTS = [8, 16, 32];
const IMAGE_DIRECTORY = join(__dirname, "..", "images");

interface Loaded {
  name: string;
  pixels: number;
  rgb: Uint8Array;
  rgba: Uint8Array;
  counts: Map<number, number>;
}

interface PaletteMeasurement {
  mse: number;
  ms: number;
}

interface Row {
  image: string;
  pixels: number;
  colors: number;
  lloyd: PaletteMeasurement;
  emeans: Map<number, PaletteMeasurement>;
}

function seedRandom(seed: number): void {
  let state = seed >>> 0;
  (Math as unknown as { random: () => number }).random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function measure(operation: () => unknown): number {
  for (let index = 0; index < WARMUPS; index += 1) {
    seedRandom(SEED);
    operation();
  }

  const samples: number[] = [];
  for (let index = 0; index < REPEATS; index += 1) {
    seedRandom(SEED);
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
  const counts = new Map<number, number>();

  for (let index = 0; index < pixels; index += 1) {
    const from = index * 4;
    const to = index * 3;
    const red = decoded.data[from];
    const green = decoded.data[from + 1];
    const blue = decoded.data[from + 2];
    rgb[to] = red;
    rgb[to + 1] = green;
    rgb[to + 2] = blue;
    rgba[from] = red;
    rgba[from + 1] = green;
    rgba[from + 2] = blue;
    rgba[from + 3] = decoded.data[from + 3];

    const key = (red << 16) | (green << 8) | blue;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  return { name, pixels, rgb, rgba, counts };
}

/**
 * Weighted MSE over distinct colours. This is exactly the mean squared nearest
 * palette distance over the pixels, because a colour that occurs `count` times
 * contributes `count` copies of the same distance.
 */
function weightedMse(image: Loaded, palette: Uint8Array): number {
  let total = 0;
  for (const [key, count] of image.counts) {
    const red = (key >> 16) & 0xff;
    const green = (key >> 8) & 0xff;
    const blue = key & 0xff;
    let best = Number.POSITIVE_INFINITY;
    for (let entry = 0; entry < palette.length; entry += 3) {
      const dr = red - palette[entry];
      const dg = green - palette[entry + 1];
      const db = blue - palette[entry + 2];
      const distance = dr * dr + dg * dg + db * db;
      if (distance < best) {
        best = distance;
      }
    }
    total += best * count;
  }
  return total / image.pixels;
}

function measurePalette(
  image: Loaded,
  colors: number,
  operation: () => Uint8Array,
): PaletteMeasurement {
  seedRandom(SEED);
  const palette = operation();
  if (palette.length !== colors * 3) {
    throw new Error(
      `${image.name} at k=${colors}: palette has ${palette.length} bytes, expected ${colors * 3}`,
    );
  }
  return {
    mse: weightedMse(image, palette),
    ms: measure(operation),
  };
}

const images = readdirSync(IMAGE_DIRECTORY)
  .filter((name) => name.endsWith(".jpg"))
  .sort()
  .map((name) => load(name));

// The RGBA entry point shares the core and differs only in the collapsed key,
// so it is checked once rather than timed on every image and colour count.
seedRandom(SEED);
const rgbaCheck = emeans_rgba(images[0].rgba, CLUSTER_COUNTS[0], REPORTED_EPOCHS);
if (rgbaCheck.length !== CLUSTER_COUNTS[0] * 4) {
  throw new Error(`emeans_rgba returned ${rgbaCheck.length} bytes`);
}

console.log("\ne-means benchmark");
console.table(
  images.map((image) => ({
    image: image.name.replace(/\.jpg$/, ""),
    pixels: image.pixels.toLocaleString("en-US"),
    distinct: image.counts.size.toLocaleString("en-US"),
    "distinct share": `${((100 * image.counts.size) / image.pixels).toFixed(1)}%`,
  })),
);

const rows: Row[] = [];
for (const image of images) {
  for (const colors of CLUSTER_COUNTS) {
    const lloyd = measurePalette(image, colors, () =>
      kmeans_rgb(image.rgb, colors, MAX_ITERATIONS, CONVERGENCE_THRESHOLD),
    );

    const emeans = new Map<number, PaletteMeasurement>();
    for (const epochs of EPOCH_VARIANTS) {
      emeans.set(
        epochs,
        measurePalette(image, colors, () => emeans_rgb(image.rgb, colors, epochs)),
      );
    }

    rows.push({
      image: image.name.replace(/\.jpg$/, ""),
      pixels: image.pixels,
      colors,
      lloyd,
      emeans,
    });
  }
}

// The seeded run is deterministic, so this is a quality regression gate rather
// than a noisy claim about one machine. The observed ratios sit at 1.00 or
// below; the margin leaves room for a deliberate schedule change, not a
// collapsed clustering.
for (const row of rows) {
  const emeans = row.emeans.get(REPORTED_EPOCHS);
  if (emeans === undefined) {
    throw new Error(`missing e-means measurement for ${REPORTED_EPOCHS} epochs`);
  }
  if (emeans.mse > row.lloyd.mse * 1.02) {
    throw new Error(
      `${row.image} at k=${row.colors}: e-means MSE ${emeans.mse.toFixed(3)} is more than 2% worse than Lloyd's ${row.lloyd.mse.toFixed(3)}`,
    );
  }
}

console.log("\nDetailed timings");
console.table(
  rows.flatMap((row) =>
    EPOCH_VARIANTS.map((epochs) => {
      const emeans = row.emeans.get(epochs)!;
      return {
        image: row.image,
        colors: row.colors,
        epochs,
        "Lloyd (ms)": Number(row.lloyd.ms.toFixed(2)),
        "e-means (ms)": Number(emeans.ms.toFixed(2)),
        "speed-up": `${(row.lloyd.ms / emeans.ms).toFixed(2)}x`,
        "Lloyd MSE": Number(row.lloyd.mse.toFixed(2)),
        "e-means MSE": Number(emeans.mse.toFixed(2)),
        "MSE ratio": (emeans.mse / row.lloyd.mse).toFixed(3),
      };
    }),
  ),
);

const reported = rows.map((row) => {
  const emeans = row.emeans.get(REPORTED_EPOCHS)!;
  return {
    image: row.image,
    pixels: row.pixels,
    colors: row.colors,
    lloydMse: row.lloyd.mse,
    emeansMse: emeans.mse,
    lloydMs: row.lloyd.ms,
    emeansMs: emeans.ms,
  };
});

console.log("\nMarkdown table");
console.log(
  "| Image | Pixels | Colors | Lloyd MSE | e-means MSE | MSE ratio | `kmeans_rgb` | `emeans_rgb` | Speed-up |",
);
console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const row of reported) {
  console.log(
    `| ${row.image} | ${row.pixels.toLocaleString("en-US")} | ${row.colors} | ${row.lloydMse.toFixed(2)} | ${row.emeansMse.toFixed(2)} | ${(row.emeansMse / row.lloydMse).toFixed(3)} | ${row.lloydMs.toFixed(2)} ms | ${row.emeansMs.toFixed(2)} ms | ${(row.lloydMs / row.emeansMs).toFixed(1)}x |`,
  );
}

console.log(
  `\nLloyd: ${MAX_ITERATIONS} iterations, convergence threshold ${CONVERGENCE_THRESHOLD}. e-means: ${REPORTED_EPOCHS} epochs and the default batch/step laws.`,
);
for (const epochs of EPOCH_VARIANTS) {
  const quality = rows.map((row) => row.emeans.get(epochs)!.mse / row.lloyd.mse);
  const speed = rows.map((row) => row.lloyd.ms / row.emeans.get(epochs)!.ms);
  console.log(
    `${epochs} epochs: median MSE ratio ${median(quality).toFixed(3)}, worst ${Math.max(...quality).toFixed(3)}; median speed-up ${median(speed).toFixed(1)}x, range ${Math.min(...speed).toFixed(1)}x to ${Math.max(...speed).toFixed(1)}x.`,
  );
}
console.log(
  `\nBoth sides were seeded from the same stream before every call, so their initial centroids match. Median of ${REPEATS} measured runs after ${WARMUPS} warm-ups.`,
);
