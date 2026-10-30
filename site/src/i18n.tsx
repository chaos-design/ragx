import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { LANGS, type Lang } from "./docs";

const STORAGE_KEY = "ragx-lang";

// UI chrome strings. Doc content itself is translated in docs/en/; this covers
// the site shell (sidebar, home, gallery, pagers, empty states).
const STRINGS = {
  zh: {
    brandSub: "文档与产物",
    navigate: "导航",
    // Mobile nav toggle (hamburger) — label reflects the action, while
    // aria-expanded carries the current state.
    navMenuOpen: "展开导航菜单",
    navMenuClose: "收起导航菜单",
    home: "首页 / 总览",
    gallery: "产物画廊",
    artifacts: "产物 (Artifacts)",
    sidebarFoot: "文档与产物在构建时内联打包；GitHub Actions 构建后经 Vercel 部署",
    langLabel: "English",
    langTitle: "切换到 English",
    // Compact label for the square language button in the mobile header.
    langShort: "EN",
    // Home
    homeCrumb: "RAGX · 在线文档中心",
    heroKicker: "Retrieval-Augmented Generation · 工程实践",
    heroTitle: "RAGX 文档中心与产物可视化",
    heroBody:
      "这是 RAGX 仓库的在线站点：左侧导航联动右侧内容区，检索 / 生成 / 质量三层文档在此渲染，仓库 docs/ 下的面试题库以内嵌可视化页面呈现。站点在构建时把 docs/ 一起打包，通过 GitHub Actions 构建并经 Vercel 直接部署。",
    readDocs: "阅读文档",
    browseArtifacts: "浏览全部产物",
    cardDocsKicker: "文档",
    cardDocsTitle: "文档中心",
    cardDocsBody: (n: number) => `检索链路、生成链路与质量评估的阅读路径，共 ${n} 篇文档。`,
    cardGalleryKicker: "产物",
    cardGalleryTitle: "产物画廊",
    cardGalleryBody: "RAGX 面试题库，以同来源 iframe 内嵌呈现。",
    cardFlowKicker: "检索",
    cardFlowTitle: "知识库创建与 Query 流程",
    cardFlowBody: "从原生知识获取、分块、嵌入、落库到检索问答的完整链路。",
    cardInterviewKicker: "可视化",
    cardInterviewTitle: "RAGX 面试题库",
    cardInterviewBody: "面向讲解与演练的交互式题库。",
    artifactsSectionTitle: "产物 (Artifacts)",
    artifactsSectionBlurb: "来自仓库 docs/，以内嵌可视化页面呈现",
    // Gallery
    galleryCrumb: "产物画廊",
    galleryTitle: "产物画廊",
    galleryDesc: (code: ReactNode) => (
      <>
        仓库 <code>{code}</code> 下的可视化产物。每个产物在独立的全屏内嵌视图中展示（同来源
        iframe，相对路径加载），可在此直接切换。
      </>
    ),
    // DocView
    docCrumb: "文档",
    docNotFound: "未找到文档",
    docNotFoundBody: (slug: ReactNode) => (
      <>
        没有 slug 为 <code>{slug}</code> 的文档。
      </>
    ),
    backHome: "返回首页",
    tocTitle: "本页目录",
    pagerPrev: "← 上一篇",
    pagerNext: "下一篇 →",
    pagerHome: "首页",
    pagerGallery: "产物画廊",
    diagramFallback: "图表渲染失败",
    // Lightbox (click-to-zoom diagram)
    diagramZoom: "图表放大",
    diagram: "图表",
    zoomIn: "放大",
    zoomOut: "缩小",
    zoomFit: "适应屏幕",
    zoomClose: "关闭",
    zoomHint: "滚轮 / 双指缩放 · 拖拽平移 · Esc 关闭",
    // ArtifactView
    artifactCrumb: "产物",
    artifactNotFound: "未找到产物",
    artifactNotFoundBody: (slug: ReactNode) => (
      <>
        没有 slug 为 <code>{slug}</code> 的产物。
      </>
    ),
    browseAll: "浏览全部产物",
    openNewWindow: "新窗口打开 ↗",
    fullscreen: "全屏沉浸 ⛶",
    escExit: "Esc 退出",
    exitImmersive: "退出 ⤢",
    immersiveSuffix: "（沉浸模式）",
    pagerPrevArtifact: "← 上一个产物",
    pagerBack: "← 返回",
    pagerNextArtifact: "下一个产物 →",
    pagerDocs: "文档中心 →",
    pagerDocsTitle: "RAGX 文档",
  },
  en: {
    brandSub: "Docs & Artifacts",
    navigate: "Navigate",
    navMenuOpen: "Open navigation menu",
    navMenuClose: "Close navigation menu",
    home: "Home",
    gallery: "Artifact Gallery",
    artifacts: "Artifacts",
    sidebarFoot: "Docs and artifacts are inlined at build time; GitHub Actions builds and Vercel deploys",
    langLabel: "中文",
    langTitle: "Switch to 中文",
    // Compact label for the square language button in the mobile header.
    langShort: "中",
    homeCrumb: "RAGX · Online Docs",
    heroKicker: "Retrieval-Augmented Generation · Engineering Practice",
    heroTitle: "RAGX Documentation & Artifact Gallery",
    heroBody:
      "This is the online site for the RAGX repository: sidebar navigation drives the content pane, docs span retrieval / generation / quality, and the interview bank under docs/ is embedded as a visual page. The site bundles docs/ at build time, builds on GitHub Actions, and deploys via Vercel.",
    readDocs: "Read the docs",
    browseArtifacts: "Browse artifacts",
    cardDocsKicker: "Docs",
    cardDocsTitle: "Documentation",
    cardDocsBody: (n: number) =>
      `Reading paths for retrieval, generation, and quality evaluation — ${n} docs in total.`,
    cardGalleryKicker: "Artifacts",
    cardGalleryTitle: "Artifact Gallery",
    cardGalleryBody: "The RAGX interview bank, embedded via a same-origin iframe.",
    cardFlowKicker: "Retrieval",
    cardFlowTitle: "Ingestion & Query Flow",
    cardFlowBody: "From raw knowledge, chunking, and embedding to retrieval-grounded Q&A.",
    cardInterviewKicker: "Visual",
    cardInterviewTitle: "RAGX Interview Bank",
    cardInterviewBody: "An interactive question bank for walkthroughs and practice.",
    artifactsSectionTitle: "Artifacts",
    artifactsSectionBlurb: "From the repo docs/, rendered as embedded visual pages",
    galleryCrumb: "Artifact Gallery",
    galleryTitle: "Artifact Gallery",
    galleryDesc: (code: ReactNode) => (
      <>
        Visual artifacts from the repo <code>{code}</code> directory. Each artifact opens in its own
        fullscreen embedded view (same-origin iframe, relative paths) and can be switched here.
      </>
    ),
    docCrumb: "Docs",
    docNotFound: "Doc not found",
    docNotFoundBody: (slug: ReactNode) => (
      <>
        No doc with slug <code>{slug}</code>.
      </>
    ),
    backHome: "Back to home",
    tocTitle: "On this page",
    pagerPrev: "← Previous",
    pagerNext: "Next →",
    pagerHome: "Home",
    pagerGallery: "Artifact Gallery",
    diagramFallback: "Diagram failed to render",
    diagramZoom: "Diagram zoom",
    diagram: "Diagram",
    zoomIn: "Zoom in",
    zoomOut: "Zoom out",
    zoomFit: "Fit to screen",
    zoomClose: "Close",
    zoomHint: "Scroll / pinch to zoom · drag to pan · Esc to close",
    artifactCrumb: "Artifact",
    artifactNotFound: "Artifact not found",
    artifactNotFoundBody: (slug: ReactNode) => (
      <>
        No artifact with slug <code>{slug}</code>.
      </>
    ),
    browseAll: "Browse all artifacts",
    openNewWindow: "Open in new window ↗",
    fullscreen: "Fullscreen ⛶",
    escExit: "Esc to exit",
    exitImmersive: "Exit ⤢",
    immersiveSuffix: " (immersive)",
    pagerPrevArtifact: "← Previous artifact",
    pagerBack: "← Back",
    pagerNextArtifact: "Next artifact →",
    pagerDocs: "Documentation →",
    pagerDocsTitle: "RAGX Docs",
  },
} as const;

export type Strings = (typeof STRINGS)["zh"];

interface LangContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: Strings;
}

const LangContext = createContext<LangContextValue | null>(null);

function initialLang(): Lang {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "zh") return saved;
  } catch {
    // localStorage unavailable (privacy mode etc.) — default below.
  }
  return "zh";
}

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);

  const setLang = (next: Lang) => {
    setLangState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Ignore persistence failures; the toggle still works for the session.
    }
  };

  // Keep <html lang> in sync for font shaping and accessibility.
  useEffect(() => {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  }, [lang]);

  const value = useMemo<LangContextValue>(() => ({ lang, setLang, t: STRINGS[lang] as Strings }), [lang]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang(): LangContextValue {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useLang must be used within LangProvider");
  return ctx;
}

/** Cycle helper used by the single toggle button. */
export function nextLang(lang: Lang): Lang {
  const i = LANGS.indexOf(lang);
  return LANGS[(i + 1) % LANGS.length];
}
