import { artifactPages } from "./artifacts";
import { docCatalog } from "./docs";
import { ROUTES } from "./routes";

const accents = ["blue", "cyan", "green", "purple", "yellow", "red"] as const;

export function Home() {
  const sections = docCatalog.sections;
  const totalDocs = docCatalog.sections.reduce((n, s) => n + s.items.length, 0);
  const firstDocSlug =
    sections[0]?.items[0]?.slug ?? Object.keys(docCatalog.index)[0] ?? "module-entrypoints";

  return (
    <div className="content-inner">
      <div className="crumb">RAGX · 在线文档中心</div>

      <div className="hero">
        <div className="hero-kicker">Retrieval-Augmented Generation · 工程实践</div>
        <h1>RAGX 文档中心与产物可视化</h1>
        <p>
          这是 RAGX 仓库的在线站点：左侧导航联动右侧内容区，检索 / 生成 / 质量三层文档
          在此渲染，仓库 <code>docs/</code> 下的面试题库以内嵌可视化页面呈现。 站点在构建时把{" "}
          <code>docs/</code> 一起打包，通过 GitHub Actions 构建并经 Vercel 直接部署。
        </p>
        <div className="hero-actions">
          <a href={ROUTES.doc(firstDocSlug)} className="btn primary">
            阅读文档
          </a>
          <a href={ROUTES.gallery} className="btn">
            浏览全部产物
          </a>
        </div>
      </div>

      <div className="grid" style={{ marginBottom: 8 }}>
        <a href={ROUTES.doc("module-entrypoints")} className="card">
          <div className="card-accent" />
          <div className="card-kicker">文档</div>
          <div className="card-title">文档中心</div>
          <div className="card-body">
            检索链路、生成链路与质量评估的阅读路径，共 {totalDocs} 篇文档。
          </div>
        </a>
        <a href={ROUTES.gallery} className="card">
          <div className="card-accent cyan" />
          <div className="card-kicker">产物</div>
          <div className="card-title">产物画廊</div>
          <div className="card-body">RAGX 面试题库，以同来源 iframe 内嵌呈现。</div>
        </a>
        <a href={ROUTES.doc("indexing-and-query-flow")} className="card">
          <div className="card-accent green" />
          <div className="card-kicker">检索</div>
          <div className="card-title">知识库创建与 Query 流程</div>
          <div className="card-body">从原生知识获取、分块、嵌入、落库到检索问答的完整链路。</div>
        </a>
        <a href={ROUTES.artifact("interview-bank")} className="card">
          <div className="card-accent purple" />
          <div className="card-kicker">可视化</div>
          <div className="card-title">RAGX 面试题库</div>
          <div className="card-body">面向讲解与演练的交互式题库。</div>
        </a>
      </div>

      {sections.map((section, i) => (
        <div key={section.id}>
          <div className="section-head">
            <h2>{section.title}</h2>
            <p>{section.blurb}</p>
          </div>
          <div className="grid">
            {section.items.map((item, j) => (
              <a key={item.slug} href={ROUTES.doc(item.slug)} className="card">
                <div className={`card-accent ${accents[(i + j) % accents.length]}`} />
                <div className="card-kicker">{section.id}</div>
                <div className="card-title">{item.title}</div>
                {item.summary ? <div className="card-body">{item.summary}</div> : null}
              </a>
            ))}
          </div>
        </div>
      ))}

      <div className="section-head">
        <h2>产物 (Artifacts)</h2>
        <p>来自仓库 docs/，以内嵌可视化页面呈现</p>
      </div>
      <div className="grid">
        {artifactPages.map((a) => (
          <a key={a.slug} href={ROUTES.artifact(a.slug)} className="card">
            <div className="card-accent" />
            <div className="card-kicker">{a.tags.join(" · ") || "artifact"}</div>
            <div className="card-title">{a.title}</div>
            <div className="card-body">{a.description}</div>
          </a>
        ))}
      </div>
    </div>
  );
}
