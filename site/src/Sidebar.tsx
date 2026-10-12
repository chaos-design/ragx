import { artifactPages } from "./artifacts";
import { catalogFor } from "./docs";
import { nextLang, useLang } from "./i18n";
import { ROUTES, type Route } from "./routes";

export function Sidebar({ route }: { route: Route }) {
  const { lang, setLang, t } = useLang();
  const catalog = catalogFor(lang);
  const activeKind = route.kind;
  const activeSlug =
    activeKind === "doc" ? route.slug : activeKind === "artifact" ? route.slug : null;

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">R</div>
        <div>
          <div className="brand-name">RAGX</div>
          <div className="brand-sub">{t.brandSub}</div>
        </div>
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

      <div className="sidebar-foot">
        {t.sidebarFoot} <code>site/dist</code>。
      </div>
    </aside>
  );
}
