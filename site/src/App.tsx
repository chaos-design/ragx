import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ArtifactView } from "./ArtifactView";
import { DocView } from "./DocView";
import { Gallery } from "./Gallery";
import { Header } from "./Header";
import { Home } from "./Home";
import { LangProvider } from "./i18n";
import { useHashRoute, type Route } from "./routes";
import { Sidebar } from "./Sidebar";
import { useBottomReserve } from "./useBottomReserve";

/** Must stay in sync with the mobile breakpoint in styles.css (860px). */
const MOBILE_NAV_QUERY = "(max-width: 860px)";

/** Stable key for the active route — used to collapse the mobile panel on navigation. */
function routeKey(route: Route): string {
  return route.kind === "doc" || route.kind === "artifact" ? `${route.kind}:${route.slug}` : route.kind;
}

function Shell() {
  const route = useHashRoute();
  // Mobile app-style layout: keep the scrollable content pane clear of the
  // fixed pager by measuring its real height (see useBottomReserve).
  useBottomReserve(route);

  const navRef = useRef<HTMLElement | null>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const [navOpen, setNavOpen] = useState(false);
  const key = routeKey(route);

  const closeNav = useCallback((restoreFocus = false) => {
    setNavOpen(false);
    if (restoreFocus) toggleRef.current?.focus();
  }, []);

  // Any navigation closes the mobile panel; focus stays on the link the user
  // just activated rather than jumping back to the toggle.
  useEffect(() => {
    setNavOpen(false);
  }, [key]);

  // Esc closes the panel and hands focus back to the toggle.
  useEffect(() => {
    if (!navOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      closeNav(true);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [navOpen, closeNav]);

  // Rotating to a desktop-width viewport must never leave the mobile panel open.
  useEffect(() => {
    const media = window.matchMedia(MOBILE_NAV_QUERY);
    const onChange = () => {
      if (!media.matches) setNavOpen(false);
    };
    onChange();
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  // Freeze the page behind the panel so the dropdown scrolls on its own.
  useEffect(() => {
    if (!navOpen) return;
    document.body.classList.add("nav-open");
    return () => document.body.classList.remove("nav-open");
  }, [navOpen]);

  // Opening moves focus into the panel, so Tab walks its items next.
  useEffect(() => {
    if (navOpen) navRef.current?.focus();
  }, [navOpen]);

  let view: ReactNode;
  switch (route.kind) {
    case "home":
      view = <Home />;
      break;
    case "doc":
      view = <DocView key={route.slug} slug={route.slug} />;
      break;
    case "artifact":
      view = <ArtifactView key={route.slug} slug={route.slug} />;
      break;
    case "gallery":
      view = <Gallery />;
      break;
    default:
      view = <Home />;
  }

  return (
    <div className="shell" data-nav-open={navOpen ? "true" : "false"}>
      <Header open={navOpen} onToggle={() => setNavOpen((open) => !open)} toggleRef={toggleRef} />
      <Sidebar ref={navRef} route={route} />
      {navOpen ? (
        <div className="nav-backdrop" onClick={() => closeNav(true)} aria-hidden="true" />
      ) : null}
      <main className="content">{view}</main>
    </div>
  );
}

export function App() {
  return (
    <LangProvider>
      <Shell />
    </LangProvider>
  );
}
