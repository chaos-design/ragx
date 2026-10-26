import { useEffect } from "react";
import { type Route } from "./routes";

const MOBILE_MEDIA_QUERY = "(max-width: 860px)";
const ROOT = "--bottom-reserve";

/**
 * On the mobile app-style layout only `.content` scrolls, while the bottom bars
 * (`.bottombar`, and `.pager` on doc/artifact pages) are `position: fixed`
 * overlays at the viewport bottom. To guarantee the last line of content is
 * never hidden behind them, we measure how high the topmost visible fixed bar
 * reaches and reserve that band on <html> as `--bottom-reserve`; `.content`
 * consumes it as its padding-bottom.
 *
 * Measuring (instead of hardcoding a bar height) keeps the reservation correct
 * whenever the pager title wraps to more lines or the language switches — both
 * change the bars' real height at runtime.
 */
export function useBottomReserve(route: Route): void {
  useEffect(() => {
    const media = window.matchMedia(MOBILE_MEDIA_QUERY);

    const visibleFixedBars = (): Element[] => {
      const out: Element[] = [];
      for (const sel of [".bottombar", ".pager"]) {
        const el = document.querySelector(sel);
        if (!el) continue;
        const cs = window.getComputedStyle(el);
        if (cs.position === "fixed" && cs.display !== "none") out.push(el);
      }
      return out;
    };

    const apply = (): void => {
      // Desktop keeps the whole-page scroll; no fixed-bar occlusion to clear.
      if (!media.matches) {
        document.documentElement.style.setProperty(ROOT, "0px");
        return;
      }
      const bars = visibleFixedBars();
      if (bars.length === 0) {
        document.documentElement.style.setProperty(ROOT, "0px");
        return;
      }
      // The topmost bar's top edge (smallest top) marks how high the overlay
      // reaches; the band below it, measured from the viewport bottom, is the
      // reservation so the last content line always sits above every fixed bar.
      const minTop = Math.min(...bars.map((el) => el.getBoundingClientRect().top));
      const reserve = Math.max(0, Math.round(window.innerHeight - minTop));
      document.documentElement.style.setProperty(ROOT, `${reserve}px`);
    };

    apply();

    // Re-measure whenever a bar's height changes (title wrap, language switch).
    const ro = new ResizeObserver(apply);
    visibleFixedBars().forEach((el) => ro.observe(el));
    window.addEventListener("resize", apply);
    media.addEventListener("change", apply);

    return () => {
      ro.disconnect();
      window.removeEventListener("resize", apply);
      media.removeEventListener("change", apply);
    };
  }, [route]);
}
