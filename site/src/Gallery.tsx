import { artifactPages } from "./artifacts";
import { ROUTES } from "./routes";

const accents = ["", "cyan", "green", "purple", "yellow", ""] as const;

export function Gallery() {
  return (
    <div className="content-inner">
      <div className="crumb">
        <a href={ROUTES.home}>RAGX</a> / 产物画廊
      </div>

      <h1 className="page-title">产物画廊</h1>
      <p className="page-desc">
        仓库 <code>docs/</code> 下的可视化产物。每个产物在独立的全屏 内嵌视图中展示（同来源
        iframe，相对路径加载），可在此直接切换。
      </p>

      <div className="grid">
        {artifactPages.map((a, i) => (
          <a key={a.slug} href={ROUTES.artifact(a.slug)} className="card">
            <div className={`card-accent ${accents[i % accents.length]}`} />
            <div className="card-kicker">{a.tags.join(" · ")}</div>
            <div className="card-title">{a.title}</div>
            <div className="card-body">{a.description}</div>
          </a>
        ))}
      </div>
    </div>
  );
}
