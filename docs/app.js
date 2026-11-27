const MAX_EDGE = 480;
const DEFAULT_PALETTE_SIZE = 16;
const MAX_ITERATIONS = 30;
const REPEATS = 3;
const SEED = 0x12345678;
const RGB_STRIDE = 3;
const RGBA_STRIDE = 4;

const panel = document.querySelector("#comparison-panel");
const title = document.querySelector("#comparison-title");
const imageInput = document.querySelector("#image-input");
const paletteSelect = document.querySelector("#palette-size");
const ditherToggle = document.querySelector("#dither-toggle");
const downloadButton = document.querySelector("#download-btn");
const paletteSwatches = document.querySelector("#palette-swatches");
const paletteCountLabel = document.querySelector("#palette-count");

const sourceCanvas = document.querySelector("#source-canvas");
const rgbCanvas = document.querySelector("#rgb-canvas");
const rgbaCanvas = document.querySelector("#rgba-canvas");
const emeansCanvas = document.querySelector("#emeans-canvas");
const skmeansCanvas = document.querySelector("#skmeans-canvas");
const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
const rgbContext = rgbCanvas.getContext("2d");
const rgbaContext = rgbaCanvas.getContext("2d");
const emeansContext = emeansCanvas.getContext("2d");
const skmeansContext = skmeansCanvas.getContext("2d");
const status = document.querySelector("#status");
const timingRows = document.querySelector("#timing-rows");
const timings = [];

let wasm = null;
let activeImage = null;
let runToken = 0;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
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

function loadImage(src = "./assets/blue-marble.jpg") {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The demo image could not be loaded."));
    image.src = src;
  });
}

function prepareImage(image) {
  const scale = Math.min(1, MAX_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));

  for (const canvas of [sourceCanvas, rgbCanvas, rgbaCanvas, emeansCanvas, skmeansCanvas]) {
    canvas.width = width;
    canvas.height = height;
  }

  sourceContext.drawImage(image, 0, 0, width, height);
  const imageData = sourceContext.getImageData(0, 0, width, height);
  // Uint8ClampedArray is accepted directly by kmeans_rgba and apply_palette.
  const rgba = imageData.data;
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

function renderPaletteResult(context, width, height, pixels, stride, palette, dithered) {
  const output = context.createImageData(width, height);
  const mapped = dithered
    ? wasm.apply_palette_dithered(pixels, width, height, palette, stride)
    : wasm.apply_palette(pixels, palette, stride);
  output.data.set(mapped);
  context.putImageData(output, 0, 0);
}

function renderAssignedResult(context, width, height, stride, palette, assignments) {
  const output = context.createImageData(width, height);

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const outputOffset = pixel * RGBA_STRIDE;
    const paletteOffset = assignments[pixel] * stride;

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

function skmeansPalette(centroids, paletteSize) {
  const palette = new Uint8Array(paletteSize * RGB_STRIDE);
  for (let index = 0; index < paletteSize; index += 1) {
    const centroid = centroids[index] ?? [0, 0, 0];
    const offset = index * RGB_STRIDE;
    for (let channel = 0; channel < RGB_STRIDE; channel += 1) {
      palette[offset + channel] = clampChannel(centroid[channel]);
    }
  }
  return palette;
}

function toHex(r, g, b) {
  return `#${[r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

function renderSwatches(palette) {
  const buttons = [];
  for (let offset = 0; offset < palette.length; offset += RGB_STRIDE) {
    const hex = toHex(palette[offset], palette[offset + 1], palette[offset + 2]);
    const swatch = document.createElement("button");
    swatch.type = "button";
    swatch.className = "swatch";
    swatch.title = hex;
    swatch.style.backgroundColor = hex;
    swatch.addEventListener("click", async () => {
      try {
        await globalThis.navigator?.clipboard?.writeText(hex);
        setStatus(`Copied ${hex} to clipboard.`);
      } catch {
        setStatus(`Palette color: ${hex}`);
      }
    });
    buttons.push(swatch);
  }
  paletteSwatches.replaceChildren(...buttons);
}

function recordTiming({ method, input, timeId, milliseconds, cost, baseline }) {
  document.querySelector(timeId).textContent = formatTime(milliseconds);
  timings.push({ method, input, milliseconds, cost, baseline });
  renderTimings();
}

function renderTimings() {
  const reference = timings.find((timing) => timing.baseline) ?? timings[0];

  timingRows.replaceChildren(
    ...timings.map(({ method, input, milliseconds, cost }) => {
      const row = document.createElement("tr");
      const ratio = reference.milliseconds / milliseconds;
      const last = cost ?? formatRatio(ratio);
      for (const [text, className] of [
        [method, "method"],
        [input, "input"],
        [formatTime(milliseconds), "time"],
        [last, "ratio"],
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

async function runComparison() {
  if (!wasm || !activeImage) return;

  const token = ++runToken;
  const paletteSize = Number(paletteSelect?.value) || DEFAULT_PALETTE_SIZE;
  const dithered = Boolean(ditherToggle?.checked);
  paletteCountLabel.textContent = `${paletteSize} colors`;
  timings.length = 0;

  const { rgb, rgba, points, width, height } = activeImage;

  setStatus("Clustering with kmeans_rgb…");
  await yieldToBrowser();
  if (token !== runToken) return;

  const rgbRun = measure(() => wasm.kmeans_rgb(rgb, paletteSize, MAX_ITERATIONS, 0.1));
  renderPaletteResult(rgbContext, width, height, rgb, RGB_STRIDE, rgbRun.result, dithered);
  renderSwatches(rgbRun.result);
  recordTiming({
    baseline: true,
    method: "kmeans_rgb",
    input: "3 × u8",
    timeId: "#rgb-time",
    milliseconds: rgbRun.milliseconds,
  });

  const mapRun = measure(() =>
    dithered
      ? wasm.apply_palette_dithered(rgb, width, height, rgbRun.result, RGB_STRIDE)
      : wasm.apply_palette(rgb, rgbRun.result, RGB_STRIDE),
  );
  recordTiming({
    method: dithered ? "apply_palette_dithered" : "apply_palette",
    input: "3 × u8",
    timeId: "#map-time",
    milliseconds: mapRun.milliseconds,
    cost: `+${formatTime(mapRun.milliseconds)} on top`,
  });

  setStatus("Clustering with kmeans_rgba…");
  await yieldToBrowser();
  if (token !== runToken) return;

  const rgbaRun = measure(() => wasm.kmeans_rgba(rgba, paletteSize, MAX_ITERATIONS, 0.1));
  renderPaletteResult(rgbaContext, width, height, rgba, RGBA_STRIDE, rgbaRun.result, dithered);
  recordTiming({
    method: "kmeans_rgba",
    input: "4 × u8",
    timeId: "#rgba-time",
    milliseconds: rgbaRun.milliseconds,
  });

  setStatus("Clustering with emeans_rgb…");
  await yieldToBrowser();
  if (token !== runToken) return;

  const emeansRun = measure(() => wasm.emeans_rgb(rgb, paletteSize));
  renderPaletteResult(emeansContext, width, height, rgb, RGB_STRIDE, emeansRun.result, dithered);
  recordTiming({
    method: "emeans_rgb",
    input: "3 × u8",
    timeId: "#emeans-time",
    milliseconds: emeansRun.milliseconds,
  });

  setStatus("Running skmeans for comparison…");
  await yieldToBrowser();
  if (token !== runToken) return;

  const skmeansRun = measure(() =>
    window.skmeans(points, paletteSize, undefined, MAX_ITERATIONS),
  );
  renderAssignedResult(
    skmeansContext,
    width,
    height,
    RGB_STRIDE,
    skmeansPalette(skmeansRun.result.centroids, paletteSize),
    skmeansRun.result.idxs,
  );
  recordTiming({
    method: "skmeans",
    input: "3 × number",
    timeId: "#skmeans-time",
    milliseconds: skmeansRun.milliseconds,
  });

  setStatus("All four results ready.");
}

async function handleImageFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const objectUrl = URL.createObjectURL(file);
  try {
    const loaded = await loadImage(objectUrl);
    title.textContent = file.name;
    activeImage = prepareImage(loaded);
    await runComparison();
  } catch (error) {
    setStatus(`Could not load image: ${error.message ?? error}`, true);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function wireControls() {
  imageInput?.addEventListener("change", (event) => {
    const file = event.target?.files?.[0];
    if (file) void handleImageFile(file);
  });

  paletteSelect?.addEventListener("change", () => {
    void runComparison();
  });

  ditherToggle?.addEventListener("change", () => {
    void runComparison();
  });

  downloadButton?.addEventListener("click", () => {
    if (typeof rgbCanvas.toDataURL !== "function") return;
    const link = document.createElement("a");
    const paletteSize = Number(paletteSelect?.value) || DEFAULT_PALETTE_SIZE;
    link.download = `quantized-${paletteSize}.png`;
    link.href = rgbCanvas.toDataURL("image/png");
    link.click?.();
  });

  panel?.addEventListener("dragover", (event) => {
    event.preventDefault();
    panel.classList.add("drag-over");
  });
  panel?.addEventListener("dragleave", () => {
    panel.classList.remove("drag-over");
  });
  panel?.addEventListener("drop", (event) => {
    event.preventDefault();
    panel.classList.remove("drag-over");
    const file = event.dataTransfer?.files?.[0];
    if (file) void handleImageFile(file);
  });
}

async function initialize() {
  try {
    wireControls();
    setStatus("Loading WebAssembly and the comparison image…");
    wasm = await import("./wasm/kmeans_wasm.js");
    await wasm.default();
    if (typeof window.skmeans !== "function") {
      throw new Error("The skmeans browser bundle did not load.");
    }

    activeImage = prepareImage(await loadImage());
    await runComparison();
  } catch (error) {
    setStatus(`Could not run the comparison: ${error.message ?? error}`, true);
  }
}

initialize();
