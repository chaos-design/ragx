import type { RefObject } from "react";
import { nextLang, useLang } from "./i18n";
import { GITHUB_REPO, ROUTES } from "./routes";

const githubIcon = (
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.2 1.77 1.2 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.4.98.02 1.97.15 2.89.4 2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.09 0 4.42-2.7 5.4-5.27 5.68.41.35.78 1.05.78 2.12 0 1.53-.01 2.77-.01 3.15 0 .31.2.67.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z" />
  </svg>
);

/**
 * Mobile-only top bar. Desktop keeps the pinned sidebar (which carries the same
 * brand, GitHub entry and language toggle), so this bar is hidden there — see
 * the `max-width: 860px` block in styles.css.
 *
 * On mobile it carries the brand (a link back to `/`), the GitHub entry, the
 * language switcher and the toggle that expands the shared sidebar into a
 * full-viewport navigation panel.
 */
export function Header({
  open,
  onToggle,
  toggleRef,
}: {
  open: boolean;
  onToggle: () => void;
  toggleRef: RefObject<HTMLButtonElement | null>;
}) {
  const { lang, setLang, t } = useLang();

  return (
    <header className="site-header">
      <a className="site-header-brand" href={ROUTES.home} aria-label={t.backHome}>
        <span className="brand-mark">R</span>
        <span className="site-header-text">
          <span className="brand-name">RAGX</span>
          <span className="brand-sub">{t.brandSub}</span>
        </span>
      </a>

      <div className="site-header-actions">
        <a
          className="header-github"
          href={GITHUB_REPO}
          target="_blank"
          rel="noreferrer"
          title="GitHub"
          aria-label="GitHub"
        >
          {githubIcon}
        </a>
        <button
          type="button"
          className="lang-toggle"
          onClick={() => setLang(nextLang(lang))}
          title={t.langTitle}
          aria-label={t.langTitle}
        >
          {t.langShort}
        </button>
        <button
          type="button"
          ref={toggleRef}
          className={`nav-toggle${open ? " open" : ""}`}
          onClick={onToggle}
          aria-expanded={open}
          aria-controls="site-nav"
          aria-label={open ? t.navMenuClose : t.navMenuOpen}
        >
          <span className="nav-toggle-bars" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
        </button>
      </div>
    </header>
  );
}
