#!/usr/bin/env node

import { copyFile, readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const outputDirectory = resolve(process.argv[2] ?? "pkg");
const readmePath = resolve("README.md");
const manifestPath = resolve(outputDirectory, "package.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

if (!manifest.name || !manifest.version || !manifest.main || !manifest.types) {
  throw new Error(`Incomplete wasm-pack manifest: ${manifestPath}`);
}

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
const exportPath = (file) => (file.startsWith("./") ? file : `./${file}`);
const licenseFiles = (await readdir(outputDirectory)).filter((file) =>
  /^LICEN[CS]E/i.test(file),
);

manifest.files = [...new Set([...(manifest.files ?? []), ...licenseFiles])];
manifest.repository = {
  type: "git",
  url: repositoryUrl,
};
manifest.bugs = {
  url: "https://github.com/vusolapohvistr/kmeans-wasm/issues",
};
manifest.exports = {
  ".": {
    types: exportPath(manifest.types),
    import: exportPath(manifest.main),
    default: exportPath(manifest.main),
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
