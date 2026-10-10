import type { ReactNode } from "react";
import { ArtifactView } from "./ArtifactView";
import { DocView } from "./DocView";
import { Gallery } from "./Gallery";
import { Home } from "./Home";
import { useHashRoute } from "./routes";
import { Sidebar } from "./Sidebar";

export function App() {
  const route = useHashRoute();

  let view: ReactNode;
  switch (route.kind) {
    case "home":
      view = <Home />;
      break;
    case "doc":
      view = <DocView slug={route.slug} />;
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
    <div className="shell">
      <Sidebar route={route} />
      <main className="content">{view}</main>
    </div>
  );
}
