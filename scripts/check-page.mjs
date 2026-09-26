#!/usr/bin/env node

// Runs docs/app.js outside a browser and asserts it actually produced images.
//
// This exists because the page had no coverage at all, and a real ImageData bug
// shipped to GitHub Pages because of it: the page read imageData.buffer,
// imageData.byteOffset and imageData.length, none of which exist on an ImageData.
// They live on the Uint8ClampedArray at imageData.data, so the buffer handed to
// kmeans_rgba was empty and every result canvas rendered black.
//
// The ImageData stub below is deliberately faithful: it exposes only what a real
// one does. Adding typed-array accessors to the stub would hide that class of bug
// again.
//
// Run with: npm run test:page   (needs `npm run pages:build` first)

import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const DOCS = resolve(import.meta.dirname, "..", "docs");
const WIDTH = 64;
const HEIGHT = 64;
const OPAQUE_ALPHA = 255;
const TRANSPARENT_ALPHA = 40;

if (!existsSync(resolve(DOCS, "wasm", "kmeans_wasm.js"))) {
  console.error("docs/wasm is missing. Run `npm run pages:build` first.");
  process.exit(1);
}

// A real ImageData has exactly these four properties. In particular it has no
// `buffer`, `byteOffset` or `length`.
function makeImageData(data, width, height) {
  return { data, width, height, colorSpace: "srgb" };
}

function makePixels(width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      data[offset] = (x * 4) & 0xff;
      data[offset + 1] = (y * 4) & 0xff;
      data[offset + 2] = ((x + y) * 2) & 0xff;
      data[offset + 3] = x < width / 2 ? OPAQUE_ALPHA : TRANSPARENT_ALPHA;
    }
  }
  return data;
}

class FakeContext {
  constructor(canvas) {
    this.canvas = canvas;
  }
  drawImage() {
    this.canvas.drewImage = true;
  }
  getImageData(_x, _y, width, height) {
    return makeImageData(makePixels(width, height), width, height);
  }
  createImageData(width, height) {
    return { data: new Uint8ClampedArray(width * height * 4), width, height };
  }
  putImageData(imageData) {
    this.canvas.painted = imageData.data;
  }
}

class FakeCanvas {
  constructor(id) {
    this.id = id;
    this.width = 0;
    this.height = 0;
    this.painted = null;
  }
  getContext() {
    this.context ??= new FakeContext(this);
    return this.context;
  }
}

class FakeClassList {
  constructor() {
    this.tokens = new Set();
  }
  toggle(token, force) {
    if (force) this.tokens.add(token);
    else this.tokens.delete(token);
  }
  contains(token) {
    return this.tokens.has(token);
  }
}

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.classList = new FakeClassList();
    this._text = "";
  }
  get textContent() {
    return this.children.length ? this.children.map((c) => c.textContent).join("") : this._text;
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  append(...nodes) {
    this.children.push(...nodes);
  }
  replaceChildren(...nodes) {
    this.children = nodes;
  }
}

const elements = new Map();
const canvases = new Map();
const ids = [
  "source-canvas",
  "rgb-canvas",
  "rgba-canvas",
  "skmeans-canvas",
  "status",
  "timing-rows",
  "rgb-time",
  "rgba-time",
  "skmeans-time",
  "source-size",
];

for (const id of ids) {
  if (id.endsWith("-canvas")) {
    const canvas = new FakeCanvas(id);
    canvases.set(id, canvas);
    elements.set(id, canvas);
  } else {
    elements.set(id, new FakeElement("div"));
  }
}

globalThis.document = {
  createElement: (tag) => new FakeElement(tag),
  querySelector: (selector) => {
    if (!selector.startsWith("#")) throw new Error(`unexpected selector ${selector}`);
    const element = elements.get(selector.slice(1));
    if (!element) throw new Error(`unknown element ${selector}`);
    return element;
  },
};
globalThis.window = {};

// The real skmeans browser bundle assigns window.skmeans when window exists.
const realFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const url = typeof input === "string" ? input : (input?.url ?? String(input));
  if (url.startsWith("file:")) {
    const bytes = await readFile(new URL(url));
    return new Response(bytes, { headers: { "content-type": "application/wasm" } });
  }
  return realFetch(input, init);
};

class FakeImage {
  constructor() {
    this.naturalWidth = WIDTH;
    this.naturalHeight = HEIGHT;
    this.onload = null;
    this.onerror = null;
  }
  set src(_value) {
    setTimeout(() => this.onload?.(), 0);
  }
}
globalThis.Image = FakeImage;

const skmeansSource = await readFile(resolve(DOCS, "vendor", "skmeans.js"), "utf8");
new Function("window", skmeansSource)(globalThis.window);

await import(pathToFileURL(resolve(DOCS, "app.js")).href);

const status = elements.get("status");
for (let attempt = 0; attempt < 400; attempt += 1) {
  if (status.textContent.includes("ready") || status.classList.contains("error")) break;
  await new Promise((resolve) => setTimeout(resolve, 50));
}

const failures = [];
const check = (condition, message) => {
  if (!condition) failures.push(message);
};

check(!status.classList.contains("error"), `page reported an error: ${status.textContent}`);
check(
  status.textContent.includes("ready"),
  `page did not finish, status was "${status.textContent}"`,
);

const distinctColours = (id) => {
  const painted = canvases.get(id).painted;
  if (!painted) return 0;
  const seen = new Set();
  for (let index = 0; index < painted.length; index += 4) {
    seen.add((painted[index] << 16) | (painted[index + 1] << 8) | painted[index + 2]);
  }
  return seen.size;
};

const alphaHistogram = (id) => {
  const painted = canvases.get(id).painted;
  if (!painted) return new Map();
  const histogram = new Map();
  for (let index = 3; index < painted.length; index += 4) {
    histogram.set(painted[index], (histogram.get(painted[index]) ?? 0) + 1);
  }
  return histogram;
};

check(canvases.get("source-canvas").drewImage === true, "the source canvas was not drawn");
check(
  canvases.get("source-canvas").width === WIDTH &&
    canvases.get("source-canvas").height === HEIGHT,
  "the source canvas was not sized to the image",
);

// The regression: an empty or all-zero palette paints a uniformly black canvas.
// The page clusters to 16 colours, so anything above one means a real palette
// was applied. Asserting an exact count would be brittle; the alpha check below
// is the exact one.
for (const id of ["rgb-canvas", "rgba-canvas", "skmeans-canvas"]) {
  check(canvases.get(id).painted !== null, `${id} was never painted`);
  check(
    distinctColours(id) > 1,
    `${id} rendered ${distinctColours(id)} distinct colours, expected a real palette`,
  );
}

// The RGBA path must cluster and render both alpha groups, not just RGB.
const rgbaAlphas = alphaHistogram("rgba-canvas");
check(
  rgbaAlphas.get(OPAQUE_ALPHA) === (WIDTH * HEIGHT) / 2 &&
    rgbaAlphas.get(TRANSPARENT_ALPHA) === (WIDTH * HEIGHT) / 2,
  `rgba-canvas alphas were ${JSON.stringify([...rgbaAlphas])}, expected two equal groups`,
);
check(
  alphaHistogram("rgb-canvas").get(TRANSPARENT_ALPHA) === undefined,
  "rgb-canvas should be fully opaque, since kmeans_rgb ignores alpha",
);

// A clustering that silently did no work reports a suspiciously zero duration.
for (const id of ["rgb-time", "rgba-time", "skmeans-time"]) {
  check(
    elements.get(id).textContent !== "—",
    `${id} was never filled in, so clustering did not run`,
  );
}

const rows = elements
  .get("timing-rows")
  .children.map((row) => row.children.map((cell) => cell.textContent));
check(rows.length === 3, `expected 3 timing rows, got ${rows.length}`);

if (failures.length > 0) {
  console.error(`\nplayground page check failed:\n${failures.map((f) => `  - ${f}`).join("\n")}`);
  console.error(`\nstatus was: ${status.textContent}`);
  for (const id of ["rgb-time", "rgba-time", "skmeans-time"]) {
    console.error(`${id}: ${elements.get(id).textContent}`);
  }
  process.exit(1);
}

console.log("playground page check passed");
console.log(`  timings: rgb ${elements.get("rgb-time").textContent}, rgba ${elements.get("rgba-time").textContent}, skmeans ${elements.get("skmeans-time").textContent}`);
for (const row of rows) console.log(`  ${row.join(" | ")}`);
