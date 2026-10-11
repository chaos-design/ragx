/// <reference types="vite/client" />

declare module "virtual:ragx-docs" {
  export const docsByLang: { zh: Record<string, string>; en: Record<string, string> };
}
