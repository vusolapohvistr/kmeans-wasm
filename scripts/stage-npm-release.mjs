#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { resolve } from "node:path";

const manifest = JSON.parse(
  await readFile(resolve("pkg/package.json"), "utf8"),
);
const cargoManifest = await readFile(resolve("Cargo.toml"), "utf8");
const cargoVersion = cargoManifest.match(/^version\s*=\s*"([^"]+)"/m)?.[1];

if (!cargoVersion || manifest.version !== cargoVersion) {
  throw new Error(
    `Generated package version ${manifest.version} does not match Cargo version ${cargoVersion ?? "unknown"}.`,
  );
}

const tag = cargoVersion.includes("-") ? "next" : "latest";
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const args = ["stage", "publish", "--tag", tag, "./pkg"];

if (process.argv.includes("--dry-run")) {
  args.push("--dry-run");
}

console.log(`Staging kmeans-wasm@${cargoVersion} with npm tag ${tag}...`);

const child = spawn(npm, args, {
  stdio: "inherit",
  env: process.env,
});

child.on("error", (error) => {
  console.error(error);
  process.exitCode = 1;
});

child.on("exit", (code, signal) => {
  if (signal) {
    console.error(`npm was terminated by ${signal}.`);
    process.exitCode = 1;
  } else {
    process.exitCode = code ?? 1;
  }
});
