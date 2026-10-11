import { artifactPages } from "./artifacts";
import { catalogFor } from "./docs";
import { useLang } from "./i18n";
import { ROUTES } from "./routes";

const accents = ["blue", "cyan", "green", "purple", "yellow", "red"] as const;

export function Home() {
  const { lang, t } = useLang();
  const sections = catalogFor(lang).sections;
  const totalDocs = sections.reduce((n, s) => n + s.items.length, 0);
  const firstDocSlug = sections[0]?.items[0]?.slug ?? "module-entrypoints";

  return (
    <div className="content-inner">
      <div className="crumb">{t.homeCrumb}</div>

      <div className="hero">
        <div className="hero-kicker">{t.heroKicker}</div>
        <h1>{t.heroTitle}</h1>
        <p>{t.heroBody}</p>
        <div className="hero-actions">
          <a href={ROUTES.doc(firstDocSlug)} className="btn primary">
            {t.readDocs}
          </a>
          <a href={ROUTES.gallery} className="btn">
            {t.browseArtifacts}
          </a>
        </div>
      </div>

      <div className="grid" style={{ marginBottom: 8 }}>
        <a href={ROUTES.doc("module-entrypoints")} className="card">
          <div className="card-accent" />
          <div className="card-kicker">{t.cardDocsKicker}</div>
          <div className="card-title">{t.cardDocsTitle}</div>
          <div className="card-body">{t.cardDocsBody(totalDocs)}</div>
        </a>
        <a href={ROUTES.gallery} className="card">
          <div className="card-accent cyan" />
          <div className="card-kicker">{t.cardGalleryKicker}</div>
          <div className="card-title">{t.cardGalleryTitle}</div>
          <div className="card-body">{t.cardGalleryBody}</div>
        </a>
        <a href={ROUTES.doc("indexing-and-query-flow")} className="card">
          <div className="card-accent green" />
          <div className="card-kicker">{t.cardFlowKicker}</div>
          <div className="card-title">{t.cardFlowTitle}</div>
          <div className="card-body">{t.cardFlowBody}</div>
        </a>
        <a href={ROUTES.artifact("interview-bank")} className="card">
          <div className="card-accent purple" />
          <div className="card-kicker">{t.cardInterviewKicker}</div>
          <div className="card-title">{t.cardInterviewTitle}</div>
          <div className="card-body">{t.cardInterviewBody}</div>
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
        <h2>{t.artifactsSectionTitle}</h2>
        <p>{t.artifactsSectionBlurb}</p>
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
