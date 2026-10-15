import { useEffect, useRef, useState } from "react";
import { artifactPages } from "./artifacts";
import { catalogFor } from "./docs";
import { nextLang, useLang } from "./i18n";
import { GITHUB_REPO, ROUTES, type Route } from "./routes";

const githubIcon = (
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.2 1.77 1.2 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.4.98.02 1.97.15 2.89.4 2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.09 0 4.42-2.7 5.4-5.27 5.68.41.35.78 1.05.78 2.12 0 1.53-.01 2.77-.01 3.15 0 .31.2.67.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z" />
  </svg>
);

// Glyphs for the mobile nav toggle: hamburger when collapsed, × when expanded.
const menuIcon = (
  <svg
    viewBox="0 0 24 24"
    aria-hidden="true"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
  >
    <path d="M4 7h16M4 12h16M4 17h16" />
  </svg>
);

const closeIcon = (
  <svg
    viewBox="0 0 24 24"
    aria-hidden="true"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
  >
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

/** Shared by the toggle's aria-controls and the collapsible menu container. */
const NAV_MENU_ID = "sidebar-nav-menu";
/** Must stay in sync with the mobile breakpoint used in styles.css. */
const MOBILE_MEDIA_QUERY = "(max-width: 860px)";

/** Stable key for the active route — used to collapse the menu after navigation. */
function routeKey(route: Route): string {
  return route.kind === "doc" || route.kind === "artifact" ? `${route.kind}:${route.slug}` : route.kind;
}

export function Sidebar({ route }: { route: Route }) {
  const { lang, setLang, t } = useLang();
  const catalog = catalogFor(lang);
  const activeKind = route.kind;
  const activeSlug =
    activeKind === "doc" ? route.slug : activeKind === "artifact" ? route.slug : null;

  // Mobile-only: the menu collapses behind the header toggle. On desktop the
  // menu is always visible, so this state is inert there.
  const [menuOpen, setMenuOpen] = useState(false);
  const sidebarRef = useRef<HTMLElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const key = routeKey(route);

  // Collapse after navigating (tapping a menu item should reveal the page).
  useEffect(() => {
    setMenuOpen(false);
  }, [key]);

  // Esc to close, click-outside to close, and drop stale state when the
  // viewport leaves the mobile breakpoint. Only bound while open.
  useEffect(() => {
    if (!menuOpen) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setMenuOpen(false);
      // Send focus back to the trigger so keyboard users keep their place.
      toggleRef.current?.focus();
    };

    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && sidebarRef.current && !sidebarRef.current.contains(target)) {
        setMenuOpen(false);
      }
    };

    // Crossing back to desktop un-hides the menu via CSS; clear the flag so
    // aria-expanded never reports an expanded menu that is not collapsible.
    const media = window.matchMedia(MOBILE_MEDIA_QUERY);
    const onBreakpointChange = () => {
      if (!media.matches) setMenuOpen(false);
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    media.addEventListener("change", onBreakpointChange);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
      media.removeEventListener("change", onBreakpointChange);
    };
  }, [menuOpen]);

  return (
    <aside className="sidebar" ref={sidebarRef}>
      {/* 顶部固定 header：不随菜单滚动 */}
      <div className="brand">
        <div className="brand-mark">R</div>
        <div>
          <div className="brand-name">RAGX</div>
          <div className="brand-sub">{t.brandSub}</div>
        </div>
        {/* Mobile only (see the max-width:860px block in styles.css): collapses
            the menu below. Hidden on desktop, where the menu is always shown. */}
        <button
          type="button"
          className="nav-toggle"
          ref={toggleRef}
          aria-expanded={menuOpen}
          aria-controls={NAV_MENU_ID}
          aria-label={menuOpen ? t.navMenuClose : t.navMenuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? closeIcon : menuIcon}
        </button>
      </div>

      <div
        id={NAV_MENU_ID}
        className={`sidebar-scroll ${menuOpen ? "is-open" : ""}`}
        // Re-tapping the already-active entry fires no hashchange, so close here
        // as well; focus is left alone to keep the default link behaviour.
        onClick={(event) => {
          if ((event.target as HTMLElement | null)?.closest?.("a")) setMenuOpen(false);
        }}
      >
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

      {/* 底部固定：GitHub 与语言切换放在一起 */}
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
