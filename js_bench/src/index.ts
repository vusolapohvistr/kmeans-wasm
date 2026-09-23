import { kmeans } from 'kmeans-wasm';
import pkg from 'kmeans-wasm/package.json';
import skmeans from 'skmeans';

console.log(pkg.version);

const MAX_ITERATIONS = 100;
const CONVERGENCE_THRESHOLD = 0.1;
const WARMUPS = 2;
const REPEATS = 10;

// Keep the data fixed and generate it outside the timed calls. This makes the
// comparison repeatable and avoids measuring random-number generation.
function generateData(size: number, dimensions: number): number[][] {
  let state = 0x12345678;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };

  return Array.from({ length: size }, () =>
    Array.from({ length: dimensions }, () => next()),
  );
}

function test<Fn extends (...args: Args) => unknown, Args extends unknown[]>(
  fn: Fn,
  times: number,
  ...args: Args
): number {
  for (let index = 0; index < WARMUPS; index += 1) {
    fn(...args);
  }

  const samples: number[] = [];
  for (let index = 0; index < times; index += 1) {
    const start = performance.now();
    fn(...args);
    samples.push(performance.now() - start);
  }

  samples.sort((left, right) => left - right);
  return samples[Math.floor(samples.length / 2)];
}

const kTests = [2, 10, 50];
const dimensionsTests = [3, 10, 50];
const dataSize = 10000;

const testsData: {
  k: number;
  dimensions: number;
  data: number[][];
}[] = kTests.flatMap((k) =>
  dimensionsTests.map((dimensions) => ({
    k,
    dimensions,
    data: generateData(dataSize, dimensions),
  })),
);

interface ITestResult {
  k: number;
  dimensions: number;
  averageTime: number;
}

const skmeansResults: ITestResult[] = testsData.map(({ k, dimensions, data }) => ({
  k,
  dimensions,
  averageTime: test(skmeans, REPEATS, data, k, undefined, MAX_ITERATIONS),
}));

const kmeansWasmResults: ITestResult[] = testsData.map(({ k, dimensions, data }) => ({
  k,
  dimensions,
  averageTime: test(
    kmeans,
    REPEATS,
    data,
    k,
    MAX_ITERATIONS,
    CONVERGENCE_THRESHOLD,
  ),
}));

console.log({ dataSize, maxIterations: MAX_ITERATIONS, convergenceThreshold: CONVERGENCE_THRESHOLD });
console.log(skmeans.name);
console.table(skmeansResults);
console.log('kmeans');
console.table(kmeansWasmResults);
