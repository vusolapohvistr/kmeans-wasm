const MAX_EDGE = 480;
const PALETTE_SIZE = 16;
const MAX_ITERATIONS = 30;
const REPEATS = 3;
const SEED = 0x12345678;
const RGB_STRIDE = 3;
const RGBA_STRIDE = 4;

const sourceCanvas = document.querySelector("#source-canvas");
const rgbCanvas = document.querySelector("#rgb-canvas");
const rgbaCanvas = document.querySelector("#rgba-canvas");
const skmeansCanvas = document.querySelector("#skmeans-canvas");
const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
const rgbContext = rgbCanvas.getContext("2d");
const rgbaContext = rgbaCanvas.getContext("2d");
const skmeansContext = skmeansCanvas.getContext("2d");
const status = document.querySelector("#status");
const timingRows = document.querySelector("#timing-rows");
const timings = [];

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function formatTime(milliseconds) {
  return milliseconds < 1
    ? `${Math.round(milliseconds * 1000)} µs`
    : `${milliseconds.toFixed(milliseconds < 10 ? 2 : 1)} ms`;
}

function formatRatio(ratio) {
  return `${ratio.toFixed(2)}×`;
}

function median(samples) {
  const sorted = [...samples].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
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

// Every run starts from the same initial centroids, so the samples measure the
// same amount of clustering work and a median stays meaningful.
function measure(operation) {
  let result;
  runWithSeed(() => {
    result = operation();
  });

  const samples = [];
  for (let run = 0; run < REPEATS; run += 1) {
    const start = performance.now();
    runWithSeed(() => {
      result = operation();
    });
    samples.push(performance.now() - start);
  }

  return { milliseconds: median(samples), result };
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

  for (const canvas of [sourceCanvas, rgbCanvas, rgbaCanvas, skmeansCanvas]) {
    canvas.width = width;
    canvas.height = height;
  }

  sourceContext.drawImage(image, 0, 0, width, height);
  const imageData = sourceContext.getImageData(0, 0, width, height);
  // Zero-copy view over the RGBA buffer. `buffer`, `byteOffset` and `length`
  // belong to the Uint8ClampedArray at `imageData.data`, not to the ImageData
  // itself, which only has `data`, `width`, `height` and `colorSpace`.
  const pixels = imageData.data;
  const rgba = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.length);
  const pixelCount = width * height;
  const rgb = new Uint8Array(pixelCount * RGB_STRIDE);
  const points = new Array(pixelCount);

  for (let index = 0; index < pixelCount; index += 1) {
    const rgbaOffset = index * RGBA_STRIDE;
    const rgbOffset = index * RGB_STRIDE;
    const point = [rgba[rgbaOffset], rgba[rgbaOffset + 1], rgba[rgbaOffset + 2]];
    rgb[rgbOffset] = point[0];
    rgb[rgbOffset + 1] = point[1];
    rgb[rgbOffset + 2] = point[2];
    points[index] = point;
  }

  document.querySelector("#source-size").textContent = `${width} × ${height}`;
  return { rgb, rgba, points, width, height };
}

function paletteCacheKey(pixels, offset, stride) {
  if (stride === RGBA_STRIDE) {
    // A packed 4-byte string keeps the cache lossless; shifting would drop alpha.
    return String.fromCharCode(
      pixels[offset],
      pixels[offset + 1],
      pixels[offset + 2],
      pixels[offset + 3],
    );
  }

  return (pixels[offset] << 16) | (pixels[offset + 1] << 8) | pixels[offset + 2];
}

function nearestPaletteOffset(pixels, offset, palette, stride) {
  let nearest = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (let paletteOffset = 0; paletteOffset < palette.length; paletteOffset += stride) {
    let distance = 0;
    for (let channel = 0; channel < stride; channel += 1) {
      const delta = pixels[offset + channel] - palette[paletteOffset + channel];
      distance += delta * delta;
    }

    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = paletteOffset;
    }
  }

  return nearest;
}

function renderPaletteResult(context, width, height, pixels, stride, palette, assignments = null) {
  const output = context.createImageData(width, height);
  const cache = new Map();

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const outputOffset = pixel * RGBA_STRIDE;
    const pixelOffset = pixel * stride;
    let paletteOffset = assignments ? assignments[pixel] * stride : undefined;

    if (paletteOffset === undefined) {
      const key = paletteCacheKey(pixels, pixelOffset, stride);
      paletteOffset = cache.get(key);
      if (paletteOffset === undefined) {
        paletteOffset = nearestPaletteOffset(pixels, pixelOffset, palette, stride);
        cache.set(key, paletteOffset);
      }
    }

    for (let channel = 0; channel < RGB_STRIDE; channel += 1) {
      output.data[outputOffset + channel] = palette[paletteOffset + channel];
    }
    output.data[outputOffset + 3] = stride === RGBA_STRIDE ? palette[paletteOffset + 3] : 255;
  }

  context.putImageData(output, 0, 0);
}

function clampChannel(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function skmeansPalette(centroids) {
  const palette = new Uint8Array(PALETTE_SIZE * RGB_STRIDE);
  for (let index = 0; index < PALETTE_SIZE; index += 1) {
    const centroid = centroids[index] ?? [0, 0, 0];
    const offset = index * RGB_STRIDE;
    for (let channel = 0; channel < RGB_STRIDE; channel += 1) {
      palette[offset + channel] = clampChannel(centroid[channel]);
    }
  }
  return palette;
}

function recordTiming({ method, input, timeId, milliseconds }) {
  document.querySelector(timeId).textContent = formatTime(milliseconds);
  timings.push({ method, input, milliseconds });
  renderTimings();
}

function renderTimings() {
  const baseline = timings[0];

  timingRows.replaceChildren(
    ...timings.map(({ method, input, milliseconds }) => {
      const row = document.createElement("tr");
      const ratio = baseline.milliseconds / milliseconds;

      for (const [text, className] of [
        [method, "method"],
        [input, "input"],
        [formatTime(milliseconds), "time"],
        [formatRatio(ratio), "ratio"],
      ]) {
        const cell = document.createElement("td");
        cell.className = className;
        cell.textContent = text;
        row.append(cell);
      }

      return row;
    }),
  );
}

async function initialize() {
  try {
    setStatus("Loading WebAssembly and the comparison image…");
    const wasm = await import("./wasm/kmeans_wasm.js");
    await wasm.default();
    if (typeof window.skmeans !== "function") {
      throw new Error("The skmeans browser bundle did not load.");
    }

    const { rgb, rgba, points, width, height } = prepareImage(await loadImage());

    setStatus("Clustering with kmeans_rgb…");
    const rgbRun = measure(() => wasm.kmeans_rgb(rgb, PALETTE_SIZE, MAX_ITERATIONS, 0.1));
    renderPaletteResult(rgbContext, width, height, rgb, RGB_STRIDE, rgbRun.result);
    recordTiming({
      method: "kmeans_rgb",
      input: "3 × u8",
      timeId: "#rgb-time",
      milliseconds: rgbRun.milliseconds,
    });

    setStatus("Clustering with kmeans_rgba…");
    const rgbaRun = measure(() => wasm.kmeans_rgba(rgba, PALETTE_SIZE, MAX_ITERATIONS, 0.1));
    renderPaletteResult(rgbaContext, width, height, rgba, RGBA_STRIDE, rgbaRun.result);
    recordTiming({
      method: "kmeans_rgba",
      input: "4 × u8",
      timeId: "#rgba-time",
      milliseconds: rgbaRun.milliseconds,
    });

    setStatus("Running skmeans for comparison…");
    const skmeansRun = measure(() =>
      window.skmeans(points, PALETTE_SIZE, undefined, MAX_ITERATIONS),
    );
    renderPaletteResult(
      skmeansContext,
      width,
      height,
      rgb,
      RGB_STRIDE,
      skmeansPalette(skmeansRun.result.centroids),
      skmeansRun.result.idxs,
    );
    recordTiming({
      method: "skmeans",
      input: "3 × number",
      timeId: "#skmeans-time",
      milliseconds: skmeansRun.milliseconds,
    });

    setStatus("All three results ready.");
  } catch (error) {
    setStatus(`Could not run the comparison: ${error.message ?? error}`, true);
  }
}

initialize();
