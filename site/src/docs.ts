// Docs are inlined at build time through the virtual module produced by
// vite.config.ts. The key is the docs-relative path without the .md extension,
// e.g. "multi-recall" (ragx ships a flat docs/ directory).
import { docs } from "virtual:ragx-docs";

export interface DocSection {
  id: string;
  title: string;
  blurb: string;
  items: DocItem[];
}

export interface DocItem {
  slug: string;
  title: string;
  summary: string;
}

export interface DocCatalog {
  sections: DocSection[];
  index: Record<string, DocItem>;
}

function heading(raw: string): string {
  const m = raw.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : "";
}

function summaryOf(raw: string): string {
  for (const line of raw.split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    return t.replace(/^[-*+]\s+/, "").slice(0, 140);
  }
  return "";
}

// Curated sections in reading order. Each `slugs` list names docs that belong
// to the section; any doc not referenced anywhere is auto-filed under a
// trailing "其他" section so no document is ever dropped from the catalog.
const sectionDefs: { id: string; title: string; blurb: string; slugs: string[] }[] = [
  {
    id: "top",
    title: "总览",
    blurb: "独立入口点与术语表，快速定位各核心模块。",
    slugs: ["module-entrypoints", "abbreviations"],
  },
  {
    id: "retrieval",
    title: "检索链路",
    blurb: "多路召回、重排、混合检索与知识库创建到 Query 检索的流程。",
    slugs: ["multi-recall", "reranking-strategies", "hybird-search", "indexing-and-query-flow"],
  },
  {
    id: "generation",
    title: "生成链路",
    blurb: "版面分析到检索生成的完整生产链路与 Prompt 组装。",
    slugs: ["layout-to-generation-flow", "prompt-assembly"],
  },
  {
    id: "quality",
    title: "质量评估",
    blurb: "检索评估设计、指标与离线 evals 门禁。",
    slugs: ["retrieval-evaluation", "evals"],
  },
];

const index: Record<string, DocItem> = {};
for (const key of Object.keys(docs).sort()) {
  const raw = docs[key];
  index[key] = { slug: key, title: heading(raw) || key, summary: summaryOf(raw) };
}

const referenced = new Set(sectionDefs.flatMap((s) => s.slugs));
const sections: DocSection[] = sectionDefs.map((def) => ({
  id: def.id,
  title: def.title,
  blurb: def.blurb,
  items: def.slugs.filter((s) => index[s]).map((s) => index[s]),
}));

// Auto-file any docs that were not explicitly listed, so the catalog is total.
const leftovers = Object.keys(index).filter((k) => !referenced.has(k));
if (leftovers.length > 0) {
  sections.push({
    id: "misc",
    title: "其他",
    blurb: "未归入上述分组的文档。",
    items: leftovers.map((k) => index[k]),
  });
}

export const docCatalog: DocCatalog = { sections, index };

/** Resolve a hash route "#/doc/<slug>" to its raw markdown. */
export function getDoc(slug: string): { raw: string; item: DocItem } | null {
  const item = index[slug];
  if (!item) return null;
  return { raw: docs[slug] ?? "", item };
}

/** Flatten slugs in reading order (section order, not alphabetical). */
export function orderedDocSlugs(): string[] {
  return docCatalog.sections.flatMap((s) => s.items.map((i) => i.slug));
}
