import { artifactPages } from "./artifacts";
import { useLang } from "./i18n";
import { ROUTES } from "./routes";

const accents = ["", "cyan", "green", "purple", "yellow", ""] as const;

export function Gallery() {
  const { t } = useLang();
  return (
    <div className="content-inner">
      <div className="crumb">
        <a href={ROUTES.home}>RAGX</a> / {t.galleryCrumb}
      </div>

      <h1 className="page-title">{t.galleryTitle}</h1>
      <p className="page-desc">{t.galleryDesc("docs/")}</p>

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
