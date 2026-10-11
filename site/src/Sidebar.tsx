import { artifactPages } from "./artifacts";
import { docCatalog } from "./docs";
import { ROUTES, type Route } from "./routes";

export function Sidebar({ route }: { route: Route }) {
  const activeKind = route.kind;
  const activeSlug =
    activeKind === "doc" ? route.slug : activeKind === "artifact" ? route.slug : null;

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">R</div>
        <div>
          <div className="brand-name">RAGX</div>
          <div className="brand-sub">Docs &amp; Artifacts</div>
        </div>
      </div>

      <nav className="nav-group">
        <div className="nav-label">Navigate</div>
        <a href={ROUTES.home} className={`nav-item ${activeKind === "home" ? "active" : ""}`}>
          <span>首页 / 总览</span>
        </a>
        <a href={ROUTES.gallery} className={`nav-item ${activeKind === "gallery" ? "active" : ""}`}>
          <span>产物画廊</span>
        </a>
      </nav>

      {docCatalog.sections.map((section) => (
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
        <div className="nav-label">产物 (Artifacts)</div>
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
        文档与产物在构建时内联打包；GitHub Actions 构建后经 Vercel 部署 <code>site/dist</code>。
      </div>
    </aside>
  );
}
