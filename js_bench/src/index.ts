import { kmeans } from 'kmeans-wasm';
import pkg from 'kmeans-wasm/package.json';
import skmeans from 'skmeans';

console.log(pkg.version);

// Replace this function with your data generation function or provide a fixed dataset
function generateData(size: number, dimensions: number): number[][] {
  return new Array(size)
    .fill(0)
    .map(() =>
      new Array(dimensions)
        .fill(0)
        .map(() => Math.random())
    );
}

function test<Fn extends (...args: Args) => Res, Args extends unknown[], Res>(fn: Fn, times: number, ...args: Args): number {
  let sum = 0;
  for (let i = 0; i < times; i++) {
    const start = performance.now();
    const res = fn(...args);
    if (res as any > 1) console.log(res);
    sum += performance.now() - start;
  }

  return sum / times;
}


const kTests = [2, 10, 50];
const dimensionsTests = [3, 10, 50];
const dataSize = 10000;
const maxIterations = 100;

const testsData: {
  k: number;
  dimensions: number;
  data: number[][];
}[] = kTests.map(k =>
  dimensionsTests.map(dimensions => {
    return {
      k,
      dimensions,
      data: generateData(dataSize, dimensions),
    };
  })
).flat();

interface ITestResult {
  k: number;
  dimensions: number;
  avarageTime: number;
}

const skmeansResults: ITestResult[] = testsData.map(({ k, dimensions, data }) => ({
  k,
  dimensions,
  avarageTime: test(skmeans, 10, data, k, undefined, maxIterations),
}));

const kmeansWasmResults: ITestResult[] = testsData.map(({ k, dimensions, data }) => ({
  k,
  dimensions,
  avarageTime: test(kmeans, 10, data, k, maxIterations),
}));

console.log({ dataSize, maxIterations });
console.log(skmeans.name);
console.table(skmeansResults);
console.log('kmeans');
console.table(kmeansWasmResults);
