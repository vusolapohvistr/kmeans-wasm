#!/usr/bin/env node

// Runs the wasm test suite in a real browser across Linux, macOS, and Windows.

import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const CACHE = join(ROOT, ".browsers");

function fail(message) {
  console.error(`\nweb tests could not run: ${message}\n`);
  process.exit(1);
}

function platformSpec() {
  if (process.platform === "win32") {
    return {
      cfpPlatform: "win64",
      chromeSubdirs: ["chrome-win64", "chrome-win"],
      chromeExe: "chrome.exe",
      driverSubdir: "chromedriver-win64",
      driverExe: "chromedriver.exe",
      playwrightCache: join(process.env.LOCALAPPDATA ?? join(homedir(), "AppData", "Local"), "ms-playwright"),
    };
  }
  if (process.platform === "darwin") {
    const cfpPlatform = process.arch === "arm64" ? "mac-arm64" : "mac-x64";
    return {
      cfpPlatform,
      chromeSubdirs: [
        join(`chrome-${cfpPlatform}`, "Google Chrome for Testing.app", "Contents", "MacOS"),
        join("chrome-mac", "Chromium.app", "Contents", "MacOS"),
      ],
      chromeExe: "Google Chrome for Testing",
      driverSubdir: `chromedriver-${cfpPlatform}`,
      driverExe: "chromedriver",
      playwrightCache: join(homedir(), "Library", "Caches", "ms-playwright"),
    };
  }
  return {
    cfpPlatform: "linux64",
    chromeSubdirs: ["chrome-linux64", "chrome-linux"],
    chromeExe: "chrome",
    driverSubdir: "chromedriver-linux64",
    driverExe: "chromedriver",
    playwrightCache: join(process.env.HOME ?? homedir(), ".cache", "ms-playwright"),
  };
}

function versionOf(binary) {
  if (!binary || !existsSync(binary)) return null;
  try {
    return execFileSync(binary, ["--version"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

function findBrowser(spec) {
  const candidates = [];
  if (process.env.CHROME) candidates.push(process.env.CHROME);
  if (existsSync(spec.playwrightCache)) {
    for (const entry of readdirSync(spec.playwrightCache).sort().reverse()) {
      if (!entry.startsWith("chromium-")) continue;
      for (const subdir of spec.chromeSubdirs) {
        candidates.push(join(spec.playwrightCache, entry, subdir, spec.chromeExe));
        if (process.platform === "darwin") {
          candidates.push(join(spec.playwrightCache, entry, subdir, "Chromium"));
        }
      }
    }
  }
  if (process.platform === "linux") {
    candidates.push("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser");
  } else if (process.platform === "darwin") {
    candidates.push("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
  } else if (process.platform === "win32") {
    candidates.push(
      "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
      "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    );
  }

  for (const candidate of candidates) {
    const reported = versionOf(candidate);
    if (reported) {
      const version = /(\d+\.\d+\.\d+\.\d+)/.exec(reported)?.[1];
      if (version) return { binary: candidate, version };
    }
    if (candidate && existsSync(candidate)) {
      console.error(`  skipping ${candidate}: it exists but does not start.`);
    }
  }
  fail(
    "no usable Chrome or Chromium.\n" +
      "  Install one with: npx playwright install chromium\n" +
      "  Or point CHROME at an existing binary.",
  );
}

async function findDriver(spec, version) {
  const cached = join(CACHE, spec.driverExe);
  if (versionOf(cached)?.includes(version)) return cached;

  const url = `https://storage.googleapis.com/chrome-for-testing-public/${version}/${spec.cfpPlatform}/${spec.driverSubdir}.zip`;
  console.log(`fetching chromedriver ${version} (${spec.cfpPlatform})`);
  const zip = join(tmpdir(), `chromedriver-${version}.zip`);
  const unpacked = join(tmpdir(), `chromedriver-${version}`);
  rmSync(unpacked, { recursive: true, force: true });
  mkdirSync(CACHE, { recursive: true });

  const response = await fetch(url);
  if (!response.ok) {
    fail(
      `could not download a chromedriver for ${version} from ${url} (HTTP ${response.status}).\n` +
        `  Install a matching chromedriver by hand and put it at ${cached}.`,
    );
  }
  const archiveBytes = new Uint8Array(await response.arrayBuffer());
  writeFileSync(zip, archiveBytes);

  if (process.platform === "win32") {
    const extract = spawnSync(
      "powershell",
      ["-NoProfile", "-Command", `Expand-Archive -Force -Path '${zip}' -DestinationPath '${unpacked}'`],
      { stdio: "inherit" },
    );
    if (extract.status !== 0) fail("could not unpack the chromedriver archive");
  } else {
    const extract = spawnSync(
      "python3",
      ["-c", "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])", zip, unpacked],
    );
    if (extract.status !== 0) fail("could not unpack the chromedriver archive");
  }

  const found = join(unpacked, spec.driverSubdir, spec.driverExe);
  if (!existsSync(found)) {
    fail(`the chromedriver archive did not contain ${spec.driverExe} (looked in ${unpacked})`);
  }
  execFileSync(
    process.execPath,
    ["-e", "const fs=require('fs');fs.copyFileSync(process.argv[1],process.argv[2]);if(process.platform!=='win32')fs.chmodSync(process.argv[2],0o755);", found, cached],
  );
  console.log(`  cached at ${cached}`);
  return cached;
}

const libs = process.env.BROWSER_LIBS;
if (libs && process.platform === "linux") {
  process.env.LD_LIBRARY_PATH = [libs, process.env.LD_LIBRARY_PATH].filter(Boolean).join(":");
}

const spec = platformSpec();
const browser = findBrowser(spec);
const driver = await findDriver(spec, browser.version);
console.log(`browser: ${browser.binary} (${browser.version})`);
console.log(`driver:  ${driver}`);

const capabilities = join(ROOT, "webdriver.json");
writeFileSync(
  capabilities,
  `${JSON.stringify(
    {
      "goog:chromeOptions": {
        binary: browser.binary,
        args: [
          "--headless=new",
          "--no-sandbox",
          "--disable-dev-shm-usage",
          "--disable-gpu",
          "--no-first-run",
        ],
      },
    },
    null,
    2,
  )}\n`,
);

const env = { ...process.env, CHROMEDRIVER: driver };
const wasmPack = process.platform === "win32" ? "wasm-pack.exe" : "wasm-pack";
const run = spawnSync(wasmPack, ["test", "--chrome", "--headless", "--chromedriver", driver], {
  cwd: ROOT,
  stdio: "inherit",
  env,
});

rmSync(capabilities, { force: true });
process.exit(run.status ?? 1);
