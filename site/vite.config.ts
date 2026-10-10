import { readdirSync, readFileSync } from "node:fs";
import { access, copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vite";

// This config lives in site/, so the site root is the directory containing it.
const siteDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(siteDir, "..");
const docsDir = path.join(repoRoot, "docs");

// Expose every docs/**/*.md file as Record<key, rawMarkdown> through a virtual
// module, so the site renders the documentation at build time and stays fully
// self-contained (no runtime fetch of the repo).
function docsModule(): Plugin {
  const virtualId = "virtual:ragx-docs";
  const resolvedId = `\0${virtualId}`;

  function collectMarkdown(dir: string, prefix: string, out: Map<string, string>) {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        collectMarkdown(full, `${prefix}${entry.name}/`, out);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        const key = `${prefix}${entry.name}`.replace(/\.md$/, "");
        out.set(key, readFileSync(full, "utf8"));
      }
    }
  }

  return {
    name: "ragx-site-docs",
    enforce: "pre",
    resolveId(id) {
      if (id === virtualId) return resolvedId;
      return null;
    },
    load(id) {
      if (id !== resolvedId) return null;
      const docs = new Map<string, string>();
      collectMarkdown(docsDir, "", docs);
      const obj: Record<string, string> = {};
      for (const [k, v] of docs) obj[k] = v;
      return `export const docs = ${JSON.stringify(obj)};`;
    },
  };
}

// The single self-contained artifact (interview-bank.html) lives inside docs/.
// Copy it into the build output after Vite has written the bundle.
function copyArtifacts(): Plugin {
  let outDir = "dist";
  const artifactSrc = path.join(docsDir, "interview-bank.html");
  return {
    name: "ragx-site-copy-artifacts",
    apply: "build",
    configResolved(config) {
      outDir = path.resolve(config.root, config.build.outDir);
    },
    async closeBundle() {
      await mkdir(outDir, { recursive: true });
      try {
        await access(artifactSrc);
        await copyFile(artifactSrc, path.join(outDir, "interview-bank.html"));
      } catch {
        // Absent artifact is not fatal for the docs build.
      }
    },
  };
}

export default defineConfig({
  // The Vite root is the site/ directory itself (index.html, src/ live here).
  root: siteDir,
  // Relative base so the built site works from a Vercel root or a nested
  // directory without a server rewrite; hash routing needs no rewrite either.
  base: "./",
  plugins: [react(), docsModule(), copyArtifacts()],
  server: {
    host: "127.0.0.1",
    port: 5175,
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
