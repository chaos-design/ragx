import { useEffect, useState } from "react";
import { artifactPages, getArtifact } from "./artifacts";
import { ROUTES } from "./routes";

export function ArtifactView({ slug }: { slug: string }) {
  const artifact = getArtifact(slug);
  const [immersive, setImmersive] = useState(false);

  // Esc exits immersive mode.
  useEffect(() => {
    if (!immersive) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setImmersive(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [immersive]);

  if (!artifact) {
    return (
      <div className="content-inner">
        <div className="crumb">产物</div>
        <h1 className="page-title">未找到产物</h1>
        <p className="page-desc">
          没有 slug 为 <code>{slug}</code> 的产物。 <a href={ROUTES.gallery}>浏览全部产物</a>。
        </p>
      </div>
    );
  }

  const at = artifactPages.findIndex((x) => x.slug === slug);
  const prev = at > 0 ? artifactPages[at - 1] : null;
  const next = at >= 0 && at < artifactPages.length - 1 ? artifactPages[at + 1] : null;

  return (
    <div className="content-inner">
      <div className="crumb">
        <a href={ROUTES.home}>RAGX</a> / 产物 / {artifact.title}
      </div>

      <h1 className="page-title">{artifact.title}</h1>
      <p className="page-desc">{artifact.description}</p>

      <div className="artifact-bar">
        <span className="pill">{artifact.tags.join(" · ")}</span>
        <span className="pill">{artifact.file}</span>
        <a className="open-link" href={artifact.file} target="_blank" rel="noreferrer">
          新窗口打开 ↗
        </a>
        <button className="btn primary" type="button" onClick={() => setImmersive(true)}>
          全屏沉浸 ⛶
        </button>
      </div>

      {!immersive ? (
        <iframe
          key={artifact.slug}
          className="artifact-frame"
          src={artifact.file}
          title={artifact.title}
        />
      ) : null}

      {immersive ? (
        <div className="immersive">
          <div className="immersive-top">
            <span className="immersive-title">{artifact.title}</span>
            <span className="pill">Esc 退出</span>
            <button className="btn" type="button" onClick={() => setImmersive(false)}>
              退出 ⤢
            </button>
          </div>
          <iframe
            className="immersive-frame"
            src={artifact.file}
            title={`${artifact.title}（沉浸模式）`}
          />
        </div>
      ) : null}

      {!immersive ? (
        <div className="pager">
          {prev ? (
            <a href={ROUTES.artifact(prev.slug)}>
              <span className="pager-kicker">← 上一个产物</span>
              <span className="pager-title">{prev.title}</span>
            </a>
          ) : (
            <a className="empty" href={ROUTES.gallery}>
              <span className="pager-kicker">← 返回</span>
              <span className="pager-title">产物画廊</span>
            </a>
          )}
          {next ? (
            <a href={ROUTES.artifact(next.slug)}>
              <span className="pager-kicker">下一个产物 →</span>
              <span className="pager-title">{next.title}</span>
            </a>
          ) : (
            <a className="empty" href={ROUTES.home}>
              <span className="pager-kicker">文档中心 →</span>
              <span className="pager-title">RAGX 文档</span>
            </a>
          )}
        </div>
      ) : null}
    </div>
  );
}
