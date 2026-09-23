const MAX_PREVIEW_EDGE = 800;
const DEFAULT_IMAGE = "./assets/blue-marble.jpg";

const $ = (selector) => document.querySelector(selector);
const sourceCanvas = $("#source-canvas");
const quantizedCanvas = $("#quantized-canvas");
const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
const quantizedContext = quantizedCanvas.getContext("2d");
const imageUpload = $("#image-upload");
const colorizeButton = $("#colorize-button");
const status = $("#status");
const paletteInput = $("#palette-size");
const iterationsInput = $("#iterations");
const thresholdInput = $("#threshold");

let currentImage;
let objectUrl;
let kmeansRgb;
let isBusy = false;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function formatTime(milliseconds) {
  if (milliseconds < 1) {
    return `${Math.round(milliseconds * 1000)} µs`;
  }

  return `${milliseconds.toFixed(milliseconds < 10 ? 2 : 1)} ms`;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function updateControlValues() {
  $("#palette-size-value").textContent = `${paletteInput.value} colors`;
  $("#iterations-value").textContent = iterationsInput.value;
  $("#threshold-value").textContent = Number(thresholdInput.value).toFixed(2);
}

function loadImageElement(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The image could not be loaded."));
    image.src = source;
  });
}

function drawSource(image) {
  const scale = Math.min(1, MAX_PREVIEW_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));

  sourceCanvas.width = width;
  sourceCanvas.height = height;
  quantizedCanvas.width = width;
  quantizedCanvas.height = height;
  sourceContext.clearRect(0, 0, width, height);
  sourceContext.drawImage(image, 0, 0, width, height);
  quantizedContext.clearRect(0, 0, width, height);
  quantizedContext.drawImage(image, 0, 0, width, height);

  $("#source-meta").textContent = `${formatNumber(width)} × ${formatNumber(height)}`;
  $("#pixels-stat").textContent = formatNumber(width * height);
  $("#result-meta").textContent = "Run to colorize";
  $("#palette-note").textContent = "Run the playground to populate";
  $("#swatches").replaceChildren();
}

function nearestPaletteColor(pixel, palette, offset) {
  const red = pixel[offset];
  const green = pixel[offset + 1];
  const blue = pixel[offset + 2];
  let nearest = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (let paletteOffset = 0; paletteOffset < palette.length; paletteOffset += 3) {
    const redDelta = red - palette[paletteOffset];
    const greenDelta = green - palette[paletteOffset + 1];
    const blueDelta = blue - palette[paletteOffset + 2];
    const distance = redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta;

    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = paletteOffset;
    }
  }

  return nearest;
}

function drawPalette(palette) {
  const swatches = $("#swatches");
  swatches.replaceChildren();

  for (let offset = 0; offset < palette.length; offset += 3) {
    const red = palette[offset].toString(16).padStart(2, "0");
    const green = palette[offset + 1].toString(16).padStart(2, "0");
    const blue = palette[offset + 2].toString(16).padStart(2, "0");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.backgroundColor = `#${red}${green}${blue}`;
    swatch.title = `#${red}${green}${blue}`;
    swatches.append(swatch);
  }

  $("#palette-note").textContent = `${palette.length / 3} representative colors`;
}

function colorize() {
  if (isBusy || !currentImage || !kmeansRgb) {
    return;
  }

  const pixelCount = sourceCanvas.width * sourceCanvas.height;
  const requestedColors = Number(paletteInput.value);
  const colorCount = Math.min(requestedColors, pixelCount);

  if (colorCount < 2) {
    setStatus("Choose an image with at least two pixels.", true);
    return;
  }

  isBusy = true;
  colorizeButton.disabled = true;
  setStatus("Clustering RGB pixels in WebAssembly…");
  $("#result-meta").textContent = "Working…";

  requestAnimationFrame(() => {
    const imageData = sourceContext.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const rgb = new Uint8Array(imageData.data);
    const clusteringStart = performance.now();
    let palette;

    try {
      palette = kmeansRgb(
        rgb,
        colorCount,
        Number(iterationsInput.value),
        Number(thresholdInput.value),
      );
    } catch (error) {
      isBusy = false;
      colorizeButton.disabled = false;
      setStatus(`Could not cluster the image: ${error.message ?? error}`, true);
      return;
    }

    const clusteringTime = performance.now() - clusteringStart;
    const mappingStart = performance.now();
    const output = quantizedContext.createImageData(sourceCanvas.width, sourceCanvas.height);
    const colorCache = new Map();

    for (let offset = 0; offset < output.data.length; offset += 4) {
      const pixelOffset = offset / 4;
      const rgbOffset = pixelOffset * 3;
      const colorKey = (rgb[rgbOffset] << 16) | (rgb[rgbOffset + 1] << 8) | rgb[rgbOffset + 2];
      let paletteOffset = colorCache.get(colorKey);

      if (paletteOffset === undefined) {
        paletteOffset = nearestPaletteColor(rgb, palette, rgbOffset);
        colorCache.set(colorKey, paletteOffset);
      }

      output.data[offset] = palette[paletteOffset];
      output.data[offset + 1] = palette[paletteOffset + 1];
      output.data[offset + 2] = palette[paletteOffset + 2];
      output.data[offset + 3] = 255;
    }

    quantizedContext.putImageData(output, 0, 0);
    const mappingTime = performance.now() - mappingStart;
    const totalTime = clusteringTime + mappingTime;

    $("#palette-stat").textContent = `${colorCount} colors`;
    $("#cluster-time-stat").textContent = formatTime(clusteringTime);
    $("#mapping-time-stat").textContent = formatTime(mappingTime);
    $("#result-meta").textContent = `${colorCount} colors · ${formatTime(totalTime)} total`;
    drawPalette(palette);

    isBusy = false;
    colorizeButton.disabled = false;
    setStatus(`Done in ${formatTime(totalTime)} — the image stayed in this browser.`);
  });
}

async function useImage(image) {
  currentImage = image;
  drawSource(image);
  colorize();
}

async function useUploadedFile(file) {
  if (!file) {
    return;
  }

  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }

  objectUrl = URL.createObjectURL(file);
  setStatus("Loading the selected image…");
  try {
    const image = await loadImageElement(objectUrl);
    await useImage(image);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function initialize() {
  updateControlValues();

  [paletteInput, iterationsInput, thresholdInput].forEach((input) => {
    input.addEventListener("input", updateControlValues);
  });
  colorizeButton.addEventListener("click", colorize);
  imageUpload.addEventListener("change", (event) => useUploadedFile(event.target.files?.[0]));

  try {
    const wasm = await import("./wasm/kmeans_wasm.js");
    await wasm.default();
    kmeansRgb = wasm.kmeans_rgb;
    setStatus("WebAssembly ready — loading the sample image…");
    const image = await loadImageElement(DEFAULT_IMAGE);
    await useImage(image);
  } catch (error) {
    setStatus(
      `Could not initialize the playground: ${error.message ?? error}. Run "npm run pages:build" before serving docs/.`,
      true,
    );
  }
}

initialize();
