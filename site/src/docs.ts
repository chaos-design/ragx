// Docs are inlined at build time through the virtual module produced by
// vite.config.ts. The key is the docs-relative path without the .md extension,
// e.g. "multi-recall" (ragx ships a flat docs/ directory). Chinese docs come
// from docs/, English translations from docs/en/ under identical keys; any
// missing English doc falls back to the Chinese source.
import { docsByLang } from "virtual:ragx-docs";

export type Lang = "zh" | "en";
export const LANGS: Lang[] = ["zh", "en"];

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
// trailing "其他/Other" section so no document is ever dropped from the catalog.
const sectionDefs: {
  id: string;
  title: Record<Lang, string>;
  blurb: Record<Lang, string>;
  slugs: string[];
}[] = [
  {
    id: "top",
    title: { zh: "总览", en: "Overview" },
    blurb: {
      zh: "独立入口点与术语表，快速定位各核心模块。",
      en: "Standalone entrypoints and the glossary for locating each core module quickly.",
    },
    slugs: ["module-entrypoints", "abbreviations"],
  },
  {
    id: "retrieval",
    title: { zh: "检索链路", en: "Retrieval Pipeline" },
    blurb: {
      zh: "多路召回、重排、混合检索与知识库创建到 Query 检索的流程。",
      en: "Multi-recall, reranking, hybrid search, and the ingestion-to-query flow.",
    },
    slugs: ["multi-recall", "reranking-strategies", "hybird-search", "indexing-and-query-flow"],
  },
  {
    id: "generation",
    title: { zh: "生成链路", en: "Generation Pipeline" },
    blurb: {
      zh: "版面分析到检索生成的完整生产链路与 Prompt 组装。",
      en: "The full layout-analysis-to-generation pipeline and prompt assembly.",
    },
    slugs: ["layout-to-generation-flow", "prompt-assembly"],
  },
  {
    id: "quality",
    title: { zh: "质量评估", en: "Quality & Evaluation" },
    blurb: {
      zh: "检索评估设计、指标与离线 evals 门禁。",
      en: "Retrieval evaluation design, metrics, and the offline evals gate.",
    },
    slugs: ["retrieval-evaluation", "evals"],
  },
];

// Raw markdown per language; missing English docs fall back to Chinese.
const rawByLang: Record<Lang, Record<string, string>> = {
  zh: docsByLang.zh,
  en: { ...docsByLang.zh, ...docsByLang.en },
};

const catalogs: Record<Lang, DocCatalog> = {} as Record<Lang, DocCatalog>;

function buildCatalog(lang: Lang): DocCatalog {
  const index: Record<string, DocItem> = {};
  for (const key of Object.keys(rawByLang[lang]).sort()) {
    const raw = rawByLang[lang][key];
    index[key] = { slug: key, title: heading(raw) || key, summary: summaryOf(raw) };
  }

  const referenced = new Set(sectionDefs.flatMap((s) => s.slugs));
  const sections: DocSection[] = sectionDefs.map((def) => ({
    id: def.id,
    title: def.title[lang],
    blurb: def.blurb[lang],
    items: def.slugs.filter((s) => index[s]).map((s) => index[s]),
  }));

  // Auto-file any docs that were not explicitly listed, so the catalog is total.
  const leftovers = Object.keys(index).filter((k) => !referenced.has(k));
  if (leftovers.length > 0) {
    sections.push({
      id: "misc",
      title: lang === "zh" ? "其他" : "Other",
      blurb: lang === "zh" ? "未归入上述分组的文档。" : "Docs not covered by the groups above.",
      items: leftovers.map((k) => index[k]),
    });
  }

  return { sections, index };
}

/** Curated catalog for a language (built lazily, then cached). */
export function catalogFor(lang: Lang): DocCatalog {
  if (!catalogs[lang]) catalogs[lang] = buildCatalog(lang);
  return catalogs[lang];
}

// Default catalog kept for convenience (Chinese source of truth).
export const docCatalog: DocCatalog = catalogFor("zh");

/** Resolve a hash route "#/doc/<slug>" to its raw markdown in the given language. */
export function getDoc(slug: string, lang: Lang = "zh"): { raw: string; item: DocItem } | null {
  const catalog = catalogFor(lang);
  const item = catalog.index[slug];
  if (!item) return null;
  return { raw: rawByLang[lang][slug] ?? "", item };
}

/** Flatten slugs in reading order (section order, not alphabetical). */
export function orderedDocSlugs(): string[] {
  return docCatalog.sections.flatMap((s) => s.items.map((i) => i.slug));
}
