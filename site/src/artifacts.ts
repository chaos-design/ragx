export interface ArtifactPage {
  slug: string;
  title: string;
  description: string;
  file: string;
  tags: string[];
}

/**
 * The single self-contained visual artifact shipped into the build output and
 * shown in a same-origin iframe. `file` is the dist-relative path (relative
 * base is "./", so it loads next to index.html).
 */
export const artifactPages: ArtifactPage[] = [
  {
    slug: "interview-bank",
    title: "RAGX 面试题库",
    description: "以 RAG 工程实践为主题的面试题库，含学习路径，可在线翻页浏览。",
    file: "interview-bank.html",
    tags: ["面试", "题库"],
  },
];

export function getArtifact(slug: string): ArtifactPage | null {
  return artifactPages.find((a) => a.slug === slug) ?? null;
}
