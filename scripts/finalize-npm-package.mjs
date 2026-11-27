#!/usr/bin/env node

import { existsSync } from "node:fs";
import { copyFile, readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const outputDirectory = resolve(process.argv[2] ?? "pkg");
const readmePath = resolve("README.md");
const manifestPath = resolve(outputDirectory, "package.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

if (!manifest.name || !manifest.version || !manifest.main || !manifest.types) {
  throw new Error(`Incomplete wasm-pack manifest: ${manifestPath}`);
}

const VALUE_EXPORTS = [
  "apply_palette",
  "apply_palette_dithered",
  "apply_palette_indices",
  "emeans",
  "emeans_f32",
  "emeans_f64",
  "emeans_rgb",
  "emeans_rgba",
  "kmeans",
  "kmeans_f32",
  "kmeans_f64",
  "kmeans_rgb",
  "kmeans_rgba",
];

const TYPE_EXPORTS = [
  "ByteBuffer",
  "IFlatKmeansResultF32",
  "IFlatKmeansResultF64",
  "IKmeansResult",
];

const exportList = VALUE_EXPORTS.join(",\n  ");
const typeList = TYPE_EXPORTS.join(",\n  ");

const nodeEntry = `/* @ts-self-types="./index.d.ts" */
import { readFileSync } from "node:fs";
import { initSync } from "./kmeans_wasm.js";

initSync({ module: readFileSync(new URL("./kmeans_wasm_bg.wasm", import.meta.url)) });

export {
  ${exportList},
} from "./kmeans_wasm.js";
`;

const browserEntry = `/* @ts-self-types="./index.d.ts" */
import init from "./kmeans_wasm.js";

await init({ module_or_path: new URL("./kmeans_wasm_bg.wasm", import.meta.url) });

export {
  ${exportList},
} from "./kmeans_wasm.js";
`;

const wasmBytes = await readFile(resolve(outputDirectory, "kmeans_wasm_bg.wasm"));
const wasmBase64 = wasmBytes.toString("base64");

const inlineEntry = `/* @ts-self-types="./inline.d.ts" */
import initWasm from "./kmeans_wasm.js";

const WASM_BASE64 = "${wasmBase64}";

function decodeBase64(base64) {
  if (typeof Uint8Array.fromBase64 === "function") {
    return Uint8Array.fromBase64(base64);
  }
  if (typeof Buffer !== "undefined") {
    return new Uint8Array(Buffer.from(base64, "base64"));
  }
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

let ready;
export default function init() {
  return (ready ??= initWasm({ module_or_path: decodeBase64(WASM_BASE64) }));
}

export {
  ${exportList},
} from "./kmeans_wasm.js";
`;

const indexDts = `export type {
  ${typeList},
} from "./kmeans_wasm.js";

export {
  ${exportList},
} from "./kmeans_wasm.js";
`;

const inlineDts = `export type {
  InitOutput,
  ${typeList},
} from "./kmeans_wasm.js";

export {
  ${exportList},
} from "./kmeans_wasm.js";

import type { InitOutput } from "./kmeans_wasm.js";
export default function init(): Promise<InitOutput>;
`;

await Promise.all([
  writeFile(resolve(outputDirectory, "node.mjs"), nodeEntry),
  writeFile(resolve(outputDirectory, "browser.mjs"), browserEntry),
  writeFile(resolve(outputDirectory, "inline.mjs"), inlineEntry),
  writeFile(resolve(outputDirectory, "index.d.ts"), indexDts),
  writeFile(resolve(outputDirectory, "inline.d.ts"), inlineDts),
]);

const collaborators = Array.isArray(manifest.collaborators)
  ? manifest.collaborators
  : [];
const [author, ...contributors] = collaborators;

if (author && !manifest.author) {
  manifest.author = author;
}

if (contributors.length > 0 && !manifest.contributors) {
  manifest.contributors = collaborators.slice(1);
}

delete manifest.collaborators;

const repositoryUrl = "git+https://github.com/vusolapohvistr/kmeans-wasm.git";
const licenseFiles = (await readdir(outputDirectory)).filter((file) =>
  /^LICEN[CS]E/i.test(file),
);
const hasSnippets = existsSync(resolve(outputDirectory, "snippets"));

manifest.main = "node.mjs";
manifest.types = "index.d.ts";
manifest.engines = {
  node: ">=20.0.0",
};
manifest.sideEffects = ["./node.mjs", "./browser.mjs"];
manifest.keywords = [
  "kmeans",
  "k-means",
  "e-means",
  "clustering",
  "color-quantization",
  "quantize",
  "palette",
  "dominant-colors",
  "image-processing",
  "canvas",
  "webassembly",
  "wasm",
  "simd",
  "machine-learning",
  "typescript",
];
manifest.files = [
  ...new Set([
    ...(manifest.files ?? []),
    "node.mjs",
    "browser.mjs",
    "inline.mjs",
    "index.d.ts",
    "inline.d.ts",
    ...licenseFiles,
    ...(hasSnippets ? ["snippets"] : []),
  ]),
];
manifest.repository = {
  type: "git",
  url: repositoryUrl,
};
manifest.bugs = {
  url: "https://github.com/vusolapohvistr/kmeans-wasm/issues",
};
manifest.exports = {
  ".": {
    node: {
      types: "./index.d.ts",
      default: "./node.mjs",
    },
    types: "./index.d.ts",
    import: "./browser.mjs",
    default: "./browser.mjs",
  },
  "./web": {
    types: "./kmeans_wasm.d.ts",
    import: "./kmeans_wasm.js",
    default: "./kmeans_wasm.js",
  },
  "./inline": {
    types: "./inline.d.ts",
    import: "./inline.mjs",
    default: "./inline.mjs",
  },
};
manifest.publishConfig = {
  ...manifest.publishConfig,
  access: "public",
};

await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

// Keep the published README in sync with the repository README. In particular,
// the RGB benchmark table is part of the npm documentation and must be visible
// from the package page.
await copyFile(readmePath, resolve(outputDirectory, "README.md"));
