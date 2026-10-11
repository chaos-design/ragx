import hljs from "highlight.js/lib/common";
import { Marked, marked } from "marked";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  securityLevel: "loose",
  maxTextSize: 250000,
  themeVariables: {
    background: "#141618",
    primaryColor: "#1b1e21",
    primaryTextColor: "#f3f0e9",
    primaryBorderColor: "#34393d",
    lineColor: "#6f9dff",
    secondaryColor: "#24282c",
    tertiaryColor: "#1b1e21",
    textFillColor: "#f3f0e9",
    mainBkg: "#1b1e21",
    nodeBorder: "#34393d",
    clusterBkg: "#1b1e21",
    clusterBorder: "#34393d",
    titleColor: "#f3f0e9",
    edgeLabelBackground: "#141618",
  },
  flowchart: { htmlLabels: false, curve: "linear" },
});

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function slugifyHeading(text: string): string {
  // GitHub-style id: lowercase, spaces -> dashes, keep CJK.
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\-_ ]/gu, "")
    .trim()
    .replace(/\s+/g, "-");
}

function stripInlineMarkdown(text: string): string {
  return text
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_~`]+/g, "")
    .trim();
}

// marked may pass a code block as a string (older) or a token object
// (v15+: { type, raw, lang, text }). Normalize both shapes.
type CoderToken = { text?: string; code?: string; lang?: string };
type HeadingToken = { depth?: number; text?: string };
type LinkToken = { href?: string; text?: string; title?: string | null };

/**
 * Rewrite a relative .md link to an in-app hash route ("#/doc/<slug>")
 * relative to the current document slug. ragx docs are flat, so links resolve
 * to a sibling slug within the docs/ directory.
 */
function resolveDocLink(href: string, fromSlug: string): string {
  if (
    href.startsWith("http://") ||
    href.startsWith("https://") ||
    href.startsWith("#") ||
    href.startsWith("mailto:")
  ) {
    return href;
  }
  if (!href.endsWith(".md")) {
    return href;
  }
  const base = fromSlug.split("/").slice(0, -1);
  const resolved: string[] = [];
  for (const part of [...base, ...href.split("/")]) {
    if (part === "" || part === ".") continue;
    if (part === "..") resolved.pop();
    else resolved.push(part);
  }
  return `#/doc/${resolved.join("/")}`;
}

export interface TocItem {
  id: string;
  text: string;
  depth: number;
}

export interface RenderedDoc {
  html: string;
  toc: TocItem[];
}

/**
 * Render markdown to a self-contained HTML string:
 *  - code blocks are highlighted; mermaid blocks become placeholders that are
 *    rendered to inline SVG afterwards,
 *  - headings get GitHub-style ids and are collected into a TOC,
 *  - relative .md links are rewritten to in-app hash routes.
 *
 * The result is fully React-safe: it is consumed through
 * `dangerouslySetInnerHTML` instead of manual DOM writes.
 */
// mermaid's render() is not safe under concurrency: it uses the id as a DOM
// element id for temporary render targets, and overlapping invocations corrupt
// the produced SVGs. Serialize all renders through a module-level queue with
// globally unique ids.
let mermaidSeq = 0;
let mermaidQueue: Promise<unknown> = Promise.resolve();

function renderMermaid(def: string): Promise<string> {
  const task = mermaidQueue.then(async () => {
    const id = `ragx-mermaid-${++mermaidSeq}`;
    const out = await mermaid.render(id, def);
    return typeof out === "string" ? out : ((out as { svg?: string }).svg ?? "");
  });
  // Keep the chain alive even when one diagram fails.
  mermaidQueue = task.then(
    () => undefined,
    () => undefined,
  );
  return task;
}

export async function renderMarkdown(markdown: string, fromSlug: string): Promise<RenderedDoc> {
  const toc: TocItem[] = [];
  const mermaids: Array<{ mid: string; def: string }> = [];
  let seq = 0;

  const parser = new Marked({ gfm: true, breaks: false });
  parser.use({
    renderer: {
      code(tokenOrStr: string | CoderToken, lang?: string) {
        const code =
          typeof tokenOrStr === "string" ? tokenOrStr : (tokenOrStr.text ?? tokenOrStr.code ?? "");
        const language = (
          lang ?? (typeof tokenOrStr === "string" ? undefined : tokenOrStr.lang)
        )?.trim();

        if (language === "mermaid") {
          const mid = `rm-${++seq}`;
          mermaids.push({ mid, def: code });
          return `<pre class="mermaid" data-mid="${mid}"></pre>`;
        }

        const cls = language ? `hljs language-${language}` : "hljs";
        const body =
          language && hljs.getLanguage(language)
            ? hljs.highlight(code, { language }).value
            : escapeHtml(code);
        return `<pre><code class="${cls}">${body}</code></pre>`;
      },
      heading(tokenOrStr: string | HeadingToken, rawText?: string, rawDepth?: string) {
        const token: HeadingToken =
          typeof tokenOrStr === "string"
            ? { text: rawText ?? "", depth: Number(rawDepth ?? 1) }
            : tokenOrStr;
        const depth = Number(token.depth ?? 1);
        const plain = stripInlineMarkdown(token.text ?? "");
        const id = slugifyHeading(plain);
        if (id && depth >= 2 && depth <= 3) {
          toc.push({ id, text: plain, depth });
        }
        const inner = marked.parseInline(token.text ?? "");
        const open = id ? `<h${depth} id="${id}">` : `<h${depth}>`;
        return `${open}${inner}</h${depth}>`;
      },
      link(tokenOrStr: string | LinkToken, rawHref?: string) {
        const token: LinkToken =
          typeof tokenOrStr === "string"
            ? { href: rawHref ?? "", text: tokenOrStr }
            : tokenOrStr;
        const next = resolveDocLink(token.href ?? "", fromSlug);
        const target = next.startsWith("#") ? "" : ` target="_blank" rel="noreferrer"`;
        const inner = marked.parseInline(token.text ?? "");
        const title = token.title ? ` title="${escapeHtml(token.title)}"` : "";
        return `<a href="${next}"${title}${target}>${inner}</a>`;
      },
    },
  });

  let html = parser.parse(markdown) as string;

  // Replace each mermaid placeholder with the rendered SVG (queued, so
  // concurrent document renders cannot collide on mermaid's DOM ids).
  for (const { mid, def } of mermaids) {
    let replacement =
      '<div class="mermaid-diagram"><div class="mermaid-fallback">图表渲染失败</div></div>';
    try {
      const svg = await renderMermaid(def);
      if (svg) {
        replacement = `<div class="mermaid-diagram">${svg}</div>`;
      }
    } catch (err) {
      console.error("mermaid render failed", err);
    }
    // Function-form replacement so `$` sequences inside the SVG are not
    // treated as replacement patterns.
    html = html.replace(`<pre class="mermaid" data-mid="${mid}"></pre>`, () => replacement);
  }

  return { html, toc };
}
