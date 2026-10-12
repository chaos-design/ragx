import { docCatalog } from "./docs";
import { GITHUB_REPO, ROUTES, type Route } from "./routes";

// The bottom bar's "文档" entry jumps to the first doc in reading order.
function firstDocSlug(): string {
  const s = docCatalog.sections[0]?.items[0]?.slug;
  return s ?? Object.keys(docCatalog.index)[0] ?? "module-entrypoints";
}

const homeIcon = (
  <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <path d="M2 7.6 8 2l6 5.6h-1.6V14H10v-3.4h-4V14H3.6V7.6H2z" />
  </svg>
);

const docIcon = (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
    <path d="M3.5 2h5.6L12 4.9V14H3.5V2z" />
    <path d="M9 2v3.2H12" />
  </svg>
);

const artifactIcon = (
  <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <rect x="2" y="2" width="5" height="5" rx="1" />
    <rect x="9" y="2" width="5" height="5" rx="1" />
    <rect x="2" y="9" width="5" height="5" rx="1" />
    <rect x="9" y="9" width="5" height="5" rx="1" />
  </svg>
);

const githubIcon = (
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.2 1.77 1.2 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.4.98.02 1.97.15 2.89.4 2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.09 0 4.42-2.7 5.4-5.27 5.68.41.35.78 1.05.78 2.12 0 1.53-.01 2.77-.01 3.15 0 .31.2.67.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z" />
  </svg>
);

export function BottomBar({ route }: { route: Route }) {
  const activeHome = route.kind === "home";
  const activeDoc = route.kind === "doc";
  const activeArtifact = route.kind === "artifact" || route.kind === "gallery";

  return (
    <nav className="bottombar" aria-label="内容切换菜单">
      <a href={ROUTES.home} className={`bottombar-item ${activeHome ? "active" : ""}`}>
        {homeIcon}
        <span>首页</span>
      </a>
      <a href={ROUTES.doc(firstDocSlug())} className={`bottombar-item ${activeDoc ? "active" : ""}`}>
        {docIcon}
        <span>文档</span>
      </a>
      <a
        href={ROUTES.gallery}
        className={`bottombar-item ${activeArtifact ? "active" : ""}`}
      >
        {artifactIcon}
        <span>产物</span>
      </a>
      <a href={GITHUB_REPO} target="_blank" rel="noreferrer" className="bottombar-item external">
        {githubIcon}
        <span>GitHub</span>
      </a>
    </nav>
  );
}
