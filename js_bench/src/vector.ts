import { kmeans, kmeans_rgb } from "kmeans-wasm";
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
const RGB_DIMENSIONS = 3;

interface VectorCase {
  points: number;
  dimensions: number;
  clusters: number;
}

interface VectorData {
  points: number[][];
  /** Only built for the three-component cases, where the packed API applies. */
  packed: Uint8Array | null;
}

interface VectorRow extends VectorCase {
  kmeansMs: number;
  packedMs: number | null;
  skmeansMs: number;
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

function makeVectorData(pointCount: number, dimensions: number): VectorData {
  const packed = dimensions === RGB_DIMENSIONS ? new Uint8Array(pointCount * dimensions) : null;
  const points = new Array<number[]>(pointCount);
  const next = makeGenerator(0x12345678);

  for (let index = 0; index < pointCount; index += 1) {
    let remaining = dimensions;
    let drawn = next();
    const point: number[] = new Array(dimensions);
    while (remaining > 0) {
      const take = Math.min(3, remaining);
      for (let byte = 0; byte < take; byte += 1) {
        point[dimensions - remaining + byte] = drawn & 0xff;
        drawn >>= 8;
      }
      remaining -= take;
      if (remaining > 0) {
        drawn = next();
      }
    }
    points[index] = point;
    packed?.set(point, index * dimensions);
  }

  return { points, packed };
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

// The result shapes are checked outside the timed calls, so a benchmark run
// cannot quietly measure an API that stopped returning clusters.
function verifyResult({ points, dimensions, clusters }: VectorCase, data: VectorData): void {
  const result = kmeans(data.points, clusters, MAX_ITERATIONS, CONVERGENCE_THRESHOLD);

  if (
    result.k !== clusters ||
    result.centroids.length !== clusters ||
    result.centroids[0]?.length !== dimensions ||
    result.idxs.length !== points
  ) {
    throw new Error(
      `Unexpected kmeans result for ${points} x ${dimensions} points and k=${clusters}.`,
    );
  }

  if (data.packed !== null) {
    const palette = kmeans_rgb(data.packed, clusters, MAX_ITERATIONS, CONVERGENCE_THRESHOLD);
    if (palette.length !== clusters * RGB_DIMENSIONS) {
      throw new Error(`Unexpected kmeans_rgb palette size for k=${clusters}.`);
    }
  }
}

function runCase(vectorCase: VectorCase): VectorRow {
  const data = makeVectorData(vectorCase.points, vectorCase.dimensions);
  verifyResult(vectorCase, data);

  const { points, dimensions, clusters } = vectorCase;
  const kmeansMs = measure(() =>
    kmeans(data.points, clusters, MAX_ITERATIONS, CONVERGENCE_THRESHOLD),
  );
  const packedMs =
    data.packed === null
      ? null
      : measure(() => kmeans_rgb(data.packed!, clusters, MAX_ITERATIONS, CONVERGENCE_THRESHOLD));
  const skmeansMs = measure(() => skmeans(data.points, clusters, RANDOM_SEEDING, MAX_ITERATIONS));

  return { points, dimensions, clusters, kmeansMs, packedMs, skmeansMs };
}

const cases: VectorCase[] = [
  { points: 1_000, dimensions: 3, clusters: 2 },
  { points: 10_000, dimensions: 3, clusters: 2 },
  { points: 10_000, dimensions: 3, clusters: 10 },
  { points: 10_000, dimensions: 3, clusters: 50 },
  { points: 100_000, dimensions: 3, clusters: 10 },
  { points: 10_000, dimensions: 10, clusters: 10 },
  { points: 10_000, dimensions: 50, clusters: 10 },
  { points: 10_000, dimensions: 50, clusters: 50 },
];

const rows = cases.map(runCase);

console.log("\nGeneral vector-space benchmark");
console.table(
  rows.map(({ points, dimensions, clusters, kmeansMs, packedMs, skmeansMs }) => ({
    points: points.toLocaleString("en-US"),
    dimensions,
    clusters,
    "kmeans (ms)": Number(kmeansMs.toFixed(2)),
    "kmeans_rgb (ms)": packedMs === null ? "—" : Number(packedMs.toFixed(2)),
    "skmeans (ms)": Number(skmeansMs.toFixed(2)),
    "vs skmeans": `${(skmeansMs / kmeansMs).toFixed(1)}x`,
  })),
);

console.log("\nMarkdown table");
console.log(
  "| Points | Dimensions | Clusters | kmeans | kmeans_rgb | skmeans | Speed-up | Packed speed-up |",
);
console.log("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const row of rows) {
  console.log(
    `| ${row.points.toLocaleString("en-US")} | ${row.dimensions} | ${row.clusters} | ${row.kmeansMs.toFixed(2)} ms | ${
      row.packedMs === null ? "—" : `${row.packedMs.toFixed(2)} ms`
    } | ${row.skmeansMs.toFixed(2)} ms | ${(row.skmeansMs / row.kmeansMs).toFixed(1)}x | ${
      row.packedMs === null ? "—" : `${(row.kmeansMs / row.packedMs).toFixed(1)}x`
    } |`,
  );
}

console.log(
  `\nMedian of ${REPEATS} runs after ${WARMUPS} warm-ups; max iterations: ${MAX_ITERATIONS}; convergence threshold: ${CONVERGENCE_THRESHOLD}.`,
);
console.log("kmeans_rgb only accepts three components, so it is measured on the three-dimensional cases.");
console.log(
  "The reference table in the README is the median of five runs of this harness; the smallest rows vary the most.",
);
