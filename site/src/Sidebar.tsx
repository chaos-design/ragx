import type { RefObject } from "react";
import { artifactPages } from "./artifacts";
import { catalogFor } from "./docs";
import { nextLang, useLang } from "./i18n";
import { GITHUB_REPO, ROUTES, type Route } from "./routes";

const githubIcon = (
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.2 1.77 1.2 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.4.98.02 1.97.15 2.89.4 2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.09 0 4.42-2.7 5.4-5.27 5.68.41.35.78 1.05.78 2.12 0 1.53-.01 2.77-.01 3.15 0 .31.2.67.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z" />
  </svg>
);

/**
 * Desktop navigation column. On mobile the same element doubles as the panel the
 * header toggle expands (hence `id="site-nav"` + `aria-controls` on the toggle,
 * and `tabIndex={-1}` so the panel itself can receive programmatic focus when
 * it opens). Its brand row and footer live in the mobile header there, so both
 * are hidden inside the mobile breakpoint.
 */
export function Sidebar({ ref, route }: { ref?: RefObject<HTMLElement | null>; route: Route }) {
  const { lang, setLang, t } = useLang();
  const catalog = catalogFor(lang);
  const activeKind = route.kind;
  const activeSlug =
    activeKind === "doc" ? route.slug : activeKind === "artifact" ? route.slug : null;

  return (
    <aside className="sidebar" id="site-nav" ref={ref} tabIndex={-1} aria-label={t.navigate}>
      {/* Pinned sidebar header row: sits above the scrollable menu, does not
          scroll. The slogan (logo + name) links back to /, so clicking it from
          any doc page returns to the overview. */}
      <div className="brand-row">
        <a href={ROUTES.home} className="brand" aria-label={t.backHome}>
          <div className="brand-mark">R</div>
          <div>
            <div className="brand-name">RAGX</div>
            <div className="brand-sub">{t.brandSub}</div>
          </div>
        </a>
      </div>

      <div className="sidebar-scroll">
        <nav className="nav-group">
          <div className="nav-label">{t.navigate}</div>
          <a href={ROUTES.home} className={`nav-item ${activeKind === "home" ? "active" : ""}`}>
            <span>{t.home}</span>
          </a>
          <a href={ROUTES.gallery} className={`nav-item ${activeKind === "gallery" ? "active" : ""}`}>
            <span>{t.gallery}</span>
          </a>
        </nav>

        {catalog.sections.map((section) => (
          <div className="nav-group" key={section.id}>
            <div className="nav-label">{section.title}</div>
            {section.items.map((item) => (
              <a
                key={item.slug}
                href={ROUTES.doc(item.slug)}
                className={`nav-item ${
                  activeKind === "doc" && activeSlug === item.slug ? "active" : ""
                }`}
                title={item.title}
              >
                <span>{item.title}</span>
              </a>
            ))}
          </div>
        ))}

        <div className="nav-group">
          <div className="nav-label">{t.artifacts}</div>
          {artifactPages.map((a) => (
            <a
              key={a.slug}
              href={ROUTES.artifact(a.slug)}
              className={`nav-item ${
                activeKind === "artifact" && activeSlug === a.slug ? "active" : ""
              }`}
            >
              <span>{a.title}</span>
            </a>
          ))}
        </div>
      </div>

      {/* 底部固定：GitHub 与语言切换放在一起（桌面侧边栏专用；移动端这两个入口
          搬到顶部 header，见 Header.tsx 与移动端断点块）。 */}
      <div className="sidebar-github">
        <a href={GITHUB_REPO} target="_blank" rel="noreferrer">
          {githubIcon}
          <span>GitHub</span>
          <span className="ext-arrow">↗</span>
        </a>
        <button
          type="button"
          className="lang-toggle"
          onClick={() => setLang(nextLang(lang))}
          title={t.langTitle}
          aria-label={t.langTitle}
        >
          {t.langLabel}
        </button>
      </div>
    </aside>
  );
}
