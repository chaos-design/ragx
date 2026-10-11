import { useEffect, useState } from "react";

export type Route =
  | { kind: "home" }
  | { kind: "doc"; slug: string }
  | { kind: "artifact"; slug: string }
  | { kind: "gallery" }
  | { kind: "notfound" };

export function parseHash(hash: string): Route {
  const clean = hash.replace(/^#\/?/, "");
  const parts = clean.split("/").filter(Boolean);
  if (parts.length === 0) return { kind: "home" };
  switch (parts[0]) {
    case "doc":
      if (parts[1]) return { kind: "doc", slug: parts.slice(1).join("/") };
      break;
    case "artifact":
      if (parts[1]) return { kind: "artifact", slug: parts[1] };
      break;
    case "artifacts":
    case "gallery":
      return { kind: "gallery" };
    default:
      return { kind: "notfound" };
  }
  return { kind: "notfound" };
}

/** All hash-route URLs the site can navigate to. */
export const ROUTES = {
  home: "#/",
  gallery: "#/artifacts",
  doc: (slug: string) => `#/doc/${slug}`,
  artifact: (slug: string) => `#/artifact/${slug}`,
} as const;

export function navigate(route: string): void {
  window.location.hash = route;
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onHash = () => {
      setRoute(parseHash(window.location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return route;
}
