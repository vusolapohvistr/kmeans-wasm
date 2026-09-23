const MAX_EDGE = 480;
const PALETTE_SIZE = 16;
const MAX_ITERATIONS = 30;
const SEED = 0x12345678;

const sourceCanvas = document.querySelector("#source-canvas");
const wasmCanvas = document.querySelector("#wasm-canvas");
const skmeansCanvas = document.querySelector("#skmeans-canvas");
const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
const wasmContext = wasmCanvas.getContext("2d");
const skmeansContext = skmeansCanvas.getContext("2d");
const status = document.querySelector("#status");

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function formatTime(milliseconds) {
  return milliseconds < 1
    ? `${Math.round(milliseconds * 1000)} µs`
    : `${milliseconds.toFixed(milliseconds < 10 ? 2 : 1)} ms`;
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function runWithSeed(operation) {
  const originalRandom = Math.random;
  Math.random = seededRandom(SEED);
  try {
    return operation();
  } finally {
    Math.random = originalRandom;
  }
}

function loadImage() {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The demo image could not be loaded."));
    image.src = "./assets/blue-marble.jpg";
  });
}

function prepareImage(image) {
  const scale = Math.min(1, MAX_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));

  for (const canvas of [sourceCanvas, wasmCanvas, skmeansCanvas]) {
    canvas.width = width;
    canvas.height = height;
  }

  sourceContext.drawImage(image, 0, 0, width, height);
  const rgba = sourceContext.getImageData(0, 0, width, height).data;
  const pixelCount = width * height;
  const rgb = new Uint8Array(pixelCount * 3);
  const points = new Array(pixelCount);

  for (let index = 0; index < pixelCount; index += 1) {
    const rgbaOffset = index * 4;
    const rgbOffset = index * 3;
    const point = [rgba[rgbaOffset], rgba[rgbaOffset + 1], rgba[rgbaOffset + 2]];
    rgb[rgbOffset] = point[0];
    rgb[rgbOffset + 1] = point[1];
    rgb[rgbOffset + 2] = point[2];
    points[index] = point;
  }

  document.querySelector("#source-size").textContent = `${width} × ${height}`;
  return { rgb, points, width, height };
}

function nearestColorIndex(pixel, palette, offset) {
  const red = pixel[offset];
  const green = pixel[offset + 1];
  const blue = pixel[offset + 2];
  let nearest = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (let paletteOffset = 0; paletteOffset < palette.length; paletteOffset += 3) {
    const dr = red - palette[paletteOffset];
    const dg = green - palette[paletteOffset + 1];
    const db = blue - palette[paletteOffset + 2];
    const distance = dr * dr + dg * dg + db * db;

    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = paletteOffset;
    }
  }

  return nearest;
}

function renderPaletteResult(context, width, height, rgba, palette, assignments = null) {
  const output = context.createImageData(width, height);
  const cache = new Map();

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const outputOffset = pixel * 4;
    const rgbOffset = pixel * 3;
    const key = (rgba[rgbOffset] << 16) | (rgba[rgbOffset + 1] << 8) | rgba[rgbOffset + 2];
    let paletteOffset = assignments ? assignments[pixel] * 3 : undefined;

    if (paletteOffset === undefined) {
      paletteOffset = cache.get(key);
      if (paletteOffset === undefined) {
        paletteOffset = nearestColorIndex(rgba, palette, rgbOffset);
        cache.set(key, paletteOffset);
      }
    }

    output.data[outputOffset] = palette[paletteOffset];
    output.data[outputOffset + 1] = palette[paletteOffset + 1];
    output.data[outputOffset + 2] = palette[paletteOffset + 2];
    output.data[outputOffset + 3] = 255;
  }

  context.putImageData(output, 0, 0);
}

function skmeansPalette(centroids) {
  const palette = new Uint8Array(PALETTE_SIZE * 3);
  for (let index = 0; index < PALETTE_SIZE; index += 1) {
    const centroid = centroids[index] ?? [0, 0, 0];
    for (let channel = 0; channel < 3; channel += 1) {
      palette[index * 3 + channel] = Math.max(0, Math.min(255, Math.round(centroid[channel])));
    }
  }
  return palette;
}

async function initialize() {
  try {
    setStatus("Loading WebAssembly and the comparison image…");
    const wasm = await import("./wasm/kmeans_wasm.js");
    await wasm.default();
    if (typeof window.skmeans !== "function") {
      throw new Error("The skmeans browser bundle did not load.");
    }

    const { rgb, points, width, height } = prepareImage(await loadImage());
    const rgba = sourceContext.getImageData(0, 0, width, height).data;

    const wasmStart = performance.now();
    const wasmPalette = runWithSeed(() => wasm.kmeans_rgb(rgb, PALETTE_SIZE, MAX_ITERATIONS, 0.1));
    renderPaletteResult(wasmContext, width, height, rgba, wasmPalette);
    const wasmTime = performance.now() - wasmStart;
    document.querySelector("#wasm-time").textContent = formatTime(wasmTime);

    setStatus("Running skmeans for comparison…");
    const skmeansStart = performance.now();
    const skmeansResult = runWithSeed(() =>
      window.skmeans(points, PALETTE_SIZE, undefined, MAX_ITERATIONS),
    );
    renderPaletteResult(
      skmeansContext,
      width,
      height,
      rgba,
      skmeansPalette(skmeansResult.centroids),
      skmeansResult.idxs,
    );
    const skmeansTime = performance.now() - skmeansStart;
    document.querySelector("#skmeans-time").textContent = formatTime(skmeansTime);

    setStatus("Both results ready.");
  } catch (error) {
    setStatus(`Could not run the comparison: ${error.message ?? error}`, true);
  }
}

initialize();
