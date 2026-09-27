#!/usr/bin/env node

// Runs the wasm test suite in a real browser.
//
// The suite in tests/web.rs cannot run natively: `js_sys` calls abort on a native
// target, so every error path and every `JsValue` result lives there and nowhere
// else. It used to be CI-only for that reason, which meant a test could sit in it
// red until a push found out.
//
// Three things are needed and none of them come with wasm-pack:
//
//   1. A browser. `npx playwright install chromium` is the easiest source, and it
//      is also the only one available here, so it is looked for first.
//   2. A chromedriver of the *exact* same version. wasm-pack downloads one, but
//      it tracks Chrome stable rather than whatever is installed, so a Playwright
//      Chromium 153 against a cached driver 154 fails with "cannot find Chrome
//      binary", which is a misleading way of saying the versions disagree. This
//      fetches the matching build from Chrome for Testing.
//   3. Capabilities, because chromedriver does not read the `CHROME` environment
//      variable and only looks for Chrome in its own well-known locations. The
//      test runner does read a `webdriver.json` in the working directory, so one
//      is written with the binary path and the headless flags.
//
// Run with: npm run test:web

import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, writeFileSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

const ROOT = resolve(import.meta.dirname, "..");
const CACHE = join(ROOT, ".browsers");
const PLAYWRIGHT_CACHE = join(process.env.HOME ?? "~", ".cache", "ms-playwright");

function fail(message) {
  console.error(`\nweb tests could not run: ${message}\n`);
  process.exit(1);
}

// Both binaries print their version and exit non-zero when a shared library is
// missing, so one helper covers the browser and the driver.
function versionOf(binary) {
  if (!binary || !existsSync(binary)) return null;
  try {
    return execFileSync(binary, ["--version"], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}

function findBrowser() {
  const candidates = [];
  if (process.env.CHROME) candidates.push(process.env.CHROME);
  // Playwright's layout has changed the directory name across versions, so both
  // are tried rather than pinning one.
  if (existsSync(PLAYWRIGHT_CACHE)) {
    for (const entry of readdirSync(PLAYWRIGHT_CACHE).sort().reverse()) {
      if (!entry.startsWith("chromium-")) continue;
      candidates.push(
        join(PLAYWRIGHT_CACHE, entry, "chrome-linux64", "chrome"),
        join(PLAYWRIGHT_CACHE, entry, "chrome-linux", "chrome"),
      );
    }
  }
  candidates.push("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser");

  for (const candidate of candidates) {
    const reported = versionOf(candidate);
    if (reported) {
      const version = /(\d+\.\d+\.\d+\.\d+)/.exec(reported)?.[1];
      if (version) return { binary: candidate, version };
    }
    if (candidate && existsSync(candidate)) {
      console.error(`  skipping ${candidate}: it exists but does not start.`);
      console.error("    That is usually a missing system library. Set BROWSER_LIBS to a");
      console.error("    directory holding them, or install them for this user.");
    }
  }
  fail(
    "no usable Chrome or Chromium.\n" +
      "  Install one with: npx playwright install chromium\n" +
      "  Or point CHROME at an existing binary.",
  );
}

function findDriver(version) {
  // A cached driver is only reused if it is the exact same build. One minor
  // version apart is enough for chromedriver to refuse.
  const cached = join(CACHE, "chromedriver");
  if (versionOf(cached)?.includes(version)) return cached;

  const url = `https://storage.googleapis.com/chrome-for-testing-public/${version}/linux64/chromedriver-linux64.zip`;
  console.log(`fetching chromedriver ${version}`);
  const zip = join(tmpdir(), `chromedriver-${version}.zip`);
  const unpacked = join(tmpdir(), `chromedriver-${version}`);
  rmSync(unpacked, { recursive: true, force: true });
  mkdirSync(CACHE, { recursive: true });

  const download = spawnSync("curl", ["-sfL", "-o", zip, url], { stdio: "inherit" });
  if (download.status !== 0) {
    fail(
      `could not download a chromedriver for ${version} from ${url}.\n` +
        "  Chrome for Testing only publishes exact versions, so an unusual\n" +
        "  build will not be there. Install a matching chromedriver by hand and\n" +
        `  put it at ${cached}.`,
    );
  }
  // There is no unzip on every machine, and python3 is already needed by the
  // other harnesses, so it does the work.
  const extract = spawnSync(
    "python3",
    ["-c", "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])", zip, unpacked],
  );
  if (extract.status !== 0) fail("could not unpack the chromedriver archive");

  const found = join(unpacked, "chromedriver-linux64", "chromedriver");
  if (!existsSync(found)) fail(`the chromedriver archive did not contain a binary (looked in ${unpacked})`);
  const bytes = execFileSync(
    "python3",
    ["-c", "import sys;sys.stdout.buffer.write(open(sys.argv[1],'rb').read())", found],
    { maxBuffer: 1 << 28 },
  );
  writeFileSync(cached, bytes);
  execFileSync("chmod", ["+x", cached]);
  console.log(`  cached at ${cached}`);
  return cached;
}

// On a machine without the system libraries Chromium needs, set BROWSER_LIBS to a
// directory holding them. This has to be applied to this process too and not only
// to the children, because the version probes below launch the browser directly
// and would otherwise all report it as broken. A CI runner that installed them
// properly has nothing to do.
const libs = process.env.BROWSER_LIBS;
if (libs) {
  process.env.LD_LIBRARY_PATH = [libs, process.env.LD_LIBRARY_PATH].filter(Boolean).join(":");
}

const browser = findBrowser();
const driver = findDriver(browser.version);
console.log(`browser: ${browser.binary} (${browser.version})`);
console.log(`driver:  ${driver}`);

// chromedriver looks for Chrome in its own well-known locations and ignores
// CHROME, but the test runner reads this file from the working directory.
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

// On a machine without the system libraries Chromium needs, set BROWSER_LIBS to a
// directory holding them and they are put on the loader path here. A CI runner
// that installed them properly has nothing to do.
const env = { ...process.env };
env.CHROMEDRIVER = driver;
if (libs) console.log(`loader path extended with ${libs}`);

const run = spawnSync("wasm-pack", ["test", "--chrome", "--headless", "--chromedriver", driver], {
  cwd: ROOT,
  stdio: "inherit",
  env,
});

rmSync(capabilities, { force: true });
process.exit(run.status ?? 1);
