import { useCallback, useEffect, useRef, useState } from "react";
import { docCatalog, getDoc, orderedDocSlugs } from "./docs";
import { Lightbox } from "./Lightbox";
import { type RenderedDoc, renderMarkdown } from "./markdown";
import { ROUTES } from "./routes";

export function DocView({ slug }: { slug: string }) {
  const result = getDoc(slug);
  const [doc, setDoc] = useState<RenderedDoc | null>(null);
  const [zoomSvg, setZoomSvg] = useState<string | null>(null);
  const [zoomTitle, setZoomTitle] = useState<string>("");
  const bodyRef = useRef<HTMLDivElement | null>(null);

  // Depend on `slug` only: `getDoc` returns a fresh object each render, and
  // re-running this effect on every render is what looped the earlier version
  // into a "stuck" state. Fetch the doc inside the effect instead.
  useEffect(() => {
    let cancelled = false;
    setDoc(null);
    const { raw } = getDoc(slug) ?? { raw: "" };
    renderMarkdown(raw, slug).then((rendered) => {
      if (cancelled) return;
      setDoc(rendered);
    });
    return () => {
      cancelled = true;
    };
  }, [slug]);

  // In-page anchors (#heading) must scroll, not rewrite the site hash route
  // (#/doc/...). Intercept them here.
  useEffect(() => {
    if (!doc) return;
    const onAnchorClick = (e: MouseEvent) => {
      const a = (e.target as HTMLElement | null)?.closest?.("a[href^='#']");
      if (!a) return;
      const href = a.getAttribute("href") ?? "";
      if (href.length <= 1 || href.startsWith("#/")) return; // site route or bare
      e.preventDefault();
      const id = decodeURIComponent(href.slice(1));
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    };
    document.addEventListener("click", onAnchorClick, true);
    return () => document.removeEventListener("click", onAnchorClick, true);
  }, [doc]);

  // Click a mermaid diagram to open it in a zoomable lightbox.
  useEffect(() => {
    if (!doc) return;
    const body = bodyRef.current;
    if (!body) return;
    const onDiagramClick = (e: MouseEvent) => {
      const target = (e.target as HTMLElement).closest?.(".mermaid-diagram");
      if (!target || !body.contains(target)) return;
      const svg = target.querySelector("svg");
      if (!svg) return;
      setZoomSvg(svg.outerHTML);
      const headings = Array.from(body.querySelectorAll("h1, h2, h3"));
      let title = "";
      for (const h of headings) {
        const pos = h.compareDocumentPosition(target);
        if (pos & Node.DOCUMENT_POSITION_FOLLOWING) {
          title = h.textContent ?? "";
        }
      }
      setZoomTitle(title || getDoc(slug)?.item.title || "图表");
    };
    body.addEventListener("click", onDiagramClick);
    return () => body.removeEventListener("click", onDiagramClick);
  }, [doc, slug]);

  const closeZoom = useCallback(() => setZoomSvg(null), []);

  if (!result) {
    return (
      <div className="content-inner">
        <div className="crumb">文档</div>
        <h1 className="page-title">未找到文档</h1>
        <p className="page-desc">
          没有 slug 为 <code>{slug}</code> 的文档。
          <a href={ROUTES.home}>返回首页</a>。
        </p>
      </div>
    );
  }

  const slugs = orderedDocSlugs();
  const i = slugs.indexOf(slug);
  const prev = i > 0 ? slugs[i - 1] : null;
  const next = i >= 0 && i < slugs.length - 1 ? slugs[i + 1] : null;
  const sectionTitle = docCatalog.sections.find((s) => s.items.some((it) => it.slug === slug))
    ?.title;

  return (
    <div className="content-inner">
      <div className="crumb">
        <a href={ROUTES.home}>RAGX</a> / {sectionTitle ?? "文档"} / {result.item.title}
      </div>

      <div className="doc-layout">
        <div>
          <div
            ref={bodyRef}
            className="doc-body"
            // Content is generated only by our own marked + hljs + mermaid pipeline
            // over repository-provided docs; no external input can reach this value.
            dangerouslySetInnerHTML={doc ? { __html: doc.html } : undefined}
          />
          {!doc ? (
            <p className="doc-loading" style={{ color: "var(--faint)" }}>
              渲染中…
            </p>
          ) : null}

          <div className="pager">
            {prev ? (
              <a href={ROUTES.doc(prev)}>
                <span className="pager-kicker">← 上一篇</span>
                <span className="pager-title">{docCatalog.index[prev]?.title ?? prev}</span>
              </a>
            ) : (
              <a className="empty" href={ROUTES.home}>
                <span className="pager-kicker">← 上一篇</span>
                <span className="pager-title">首页</span>
              </a>
            )}
            {next ? (
              <a href={ROUTES.doc(next)}>
                <span className="pager-kicker">下一篇 →</span>
                <span className="pager-title">{docCatalog.index[next]?.title ?? next}</span>
              </a>
            ) : (
              <a className="empty" href={ROUTES.gallery}>
                <span className="pager-kicker">下一篇 →</span>
                <span className="pager-title">产物画廊</span>
              </a>
            )}
          </div>
        </div>

        {doc && doc.toc.length > 0 ? (
          <div className="doc-toc">
            <h4>本页目录</h4>
            <ol>
              {doc.toc.map((t) => (
                <li key={t.id} style={{ paddingLeft: t.depth === 3 ? 14 : 0 }}>
                  <a href={`#${t.id}`}>{t.text}</a>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </div>

      {zoomSvg ? <Lightbox svgHtml={zoomSvg} title={zoomTitle} onClose={closeZoom} /> : null}
    </div>
  );
}
