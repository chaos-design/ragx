import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";
import "highlight.js/styles/atom-one-dark.css";

const container = document.getElementById("root");
if (!container) {
  throw new Error("RAGX site: #root element is missing in index.html");
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
