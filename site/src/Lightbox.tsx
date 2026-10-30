import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useLang } from "./i18n";

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}

const MIN_SCALE = 0.05;
const MAX_SCALE = 8;
/** Breathing room (css px) kept around the diagram when fitting it to the
 *  viewport, so the fitted diagram never touches the edges. */
const FIT_MARGIN = 16;
/** How far a diagram smaller than the viewport may be dragged off-centre. */
const PAN_SLACK = 48;

interface Size {
  w: number;
  h: number;
}

interface View {
  /** Diagram scale; 1 means one viewBox unit renders as one css px. */
  s: number;
  /** Pan offset in screen px, applied after scaling. */
  x: number;
  y: number;
}

/**
 * Click-to-zoom lightbox for document diagrams.
 *
 * The diagram is a mermaid SVG whose `viewBox` describes its true size. We give
 * the stage that intrinsic size in css px and drive zoom purely through
 * `transform: scale()`, which means:
 *   - opening the lightbox scales the diagram to *fit* the viewport (contain),
 *     so a wide flowchart fills the screen instead of rendering at container
 *     width and wasting most of the vertical space;
 *   - zoom is anchored at the pointer/pinch centre, so the point you look at
 *     stays put;
 *   - pan is clamped, so the diagram can never be dragged out of sight.
 *
 * Wheel / pinch / buttons all funnel into the same `zoomAt`, and "fit" is
 * recomputed from the live viewport box on open and on resize.
 */
export function Lightbox({
  svgHtml,
  title,
  onClose,
}: {
  svgHtml: string;
  title?: string;
  onClose: () => void;
}) {
  const { t } = useLang();
  const boxRef = useRef<HTMLDivElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);

  // Intrinsic diagram size, read from the SVG's viewBox after mount.
  const [size, setSize] = useState<Size | null>(null);
  // Latest view kept in a ref so gesture handlers always read fresh values
  // without re-subscribing on every frame.
  const viewRef = useRef<View>({ s: 1, x: 0, y: 0 });
  const [view, setView] = useState<View>(viewRef.current);

  const commit = useCallback((next: View) => {
    viewRef.current = next;
    setView(next);
  }, []);

  /** Clamp pan so at least a sliver of the diagram stays on screen. */
  const clampPan = useCallback((next: View): View => {
    const vp = viewportRef.current;
    if (!vp) return next;
    const overX = Math.max(0, (next.s * (size?.w ?? 0) - vp.clientWidth) / 2);
    const overY = Math.max(0, (next.s * (size?.h ?? 0) - vp.clientHeight) / 2);
    return {
      s: next.s,
      x: clamp(next.x, -(overX + PAN_SLACK), overX + PAN_SLACK),
      y: clamp(next.y, -(overY + PAN_SLACK), overY + PAN_SLACK),
    };
  }, [size]);

  /** Scale that makes the whole diagram fit inside the viewport ("contain"). */
  const computeFit = useCallback((): number => {
    const vp = viewportRef.current;
    if (!vp || !size || size.w <= 0 || size.h <= 0) return 1;
    const availW = Math.max(1, vp.clientWidth - FIT_MARGIN * 2);
    const availH = Math.max(1, vp.clientHeight - FIT_MARGIN * 2);
    return clamp(Math.min(availW / size.w, availH / size.h), MIN_SCALE, MAX_SCALE);
  }, [size]);

  const applyFit = useCallback(() => {
    commit({ s: computeFit(), x: 0, y: 0 });
  }, [commit, computeFit]);

  /** Zoom by `factor`, keeping the screen point (cx, cy) anchored. */
  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) => {
      const vp = viewportRef.current;
      if (!vp) return;
      const rect = vp.getBoundingClientRect();
      // Anchor position relative to the viewport centre (transform-origin).
      const ax = cx - rect.left - rect.width / 2;
      const ay = cy - rect.top - rect.height / 2;
      const v = viewRef.current;
      const s2 = clamp(v.s * factor, MIN_SCALE, MAX_SCALE);
      const k = s2 / v.s;
      // Solve x' = a - k(a - x) so the anchor point stays fixed on screen.
      commit(clampPan({ s: s2, x: ax - k * (ax - v.x), y: ay - k * (ay - v.y) }));
    },
    [clampPan, commit],
  );

  // ---- Mount: normalise the SVG so it can be scaled by the stage box -------
  useLayoutEffect(() => {
    const stage = stageRef.current;
    const svg = stage?.querySelector("svg");
    if (!stage || !svg) return;

    const vb = svg.viewBox?.baseVal;
    const rect = svg.getBoundingClientRect();
    const w = vb && vb.width > 0 ? vb.width : rect.width || 800;
    const h = vb && vb.height > 0 ? vb.height : rect.height || 600;

    // Mermaid ships `width="100%"` + an inline `max-width:<intrinsic>px`, which
    // makes the SVG track its container instead of the stage box. Drop both so
    // the stage controls the size and the transform controls the zoom.
    svg.style.maxWidth = "none";
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.style.display = "block";
    svg.removeAttribute("height");
    svg.setAttribute("width", "100%");

    setSize({ w, h });
  }, [svgHtml]);

  // Fit as soon as the intrinsic size is known (and re-fit on rotation/resize
  // while the diagram has not been manually zoomed yet).
  useLayoutEffect(() => {
    if (!size) return;
    commit({ s: computeFit(), x: 0, y: 0 });
  }, [size, commit, computeFit]);

  // ---- Keyboard, scroll lock, focus ---------------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const prevFocus = document.activeElement as HTMLElement | null;
    boxRef.current?.focus({ preventScroll: true });
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      prevFocus?.focus?.({ preventScroll: true });
    };
  }, [onClose]);

  // Keep the fitted diagram centred when the viewport changes size (mobile
  // browser chrome collapsing, device rotation, …).
  useEffect(() => {
    const vp = viewportRef.current;
    if (!vp) return;
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const v = viewRef.current;
        const fit = computeFit();
        // Only re-fit if the user has not zoomed away from the fitted scale;
        // otherwise just re-clamp the pan against the new box.
        const wasFitted = Math.abs(v.s - fit) < 0.02 || v.s <= fit;
        commit(wasFitted ? { s: fit, x: 0, y: 0 } : clampPan(v));
      });
    });
    ro.observe(vp);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [clampPan, commit, computeFit]);

  // Wheel needs a non-passive listener to preventDefault; React's onWheel is
  // passive in some browsers, which let the page scroll behind the lightbox.
  useEffect(() => {
    const vp = viewportRef.current;
    if (!vp) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 0.05 : 0.0022;
      zoomAt(Math.exp(-e.deltaY * unit), e.clientX, e.clientY);
    };
    vp.addEventListener("wheel", onWheel, { passive: false });
    return () => vp.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  // ---- Pointer gestures: drag to pan, two fingers to pinch-zoom -----------
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const pinchRef = useRef<{ dist: number; x: number; y: number } | null>(null);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    // Capture keeps the drag alive when the finger leaves the viewport. It can
    // throw if the pointer is no longer active (fast taps, synthetic events),
    // which must not break the gesture bookkeeping below.
    try {
      (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    } catch {
      /* not capturable — panning still works while inside the viewport */
    }
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 1) {
      dragRef.current = { x: e.clientX, y: e.clientY };
    } else if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      pinchRef.current = {
        dist: Math.hypot(a.x - b.x, a.y - b.y),
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
      };
      dragRef.current = null;
    }
  }, []);

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!pointers.current.has(e.pointerId)) return;
      pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (pointers.current.size >= 2) {
        const [a, b] = [...pointers.current.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
        const prev = pinchRef.current;
        pinchRef.current = { dist, x: mid.x, y: mid.y };
        if (!prev || prev.dist <= 0) return;
        const vp = viewportRef.current;
        if (!vp) return;
        const rect = vp.getBoundingClientRect();
        const v = viewRef.current;
        const s2 = clamp(v.s * (dist / prev.dist), MIN_SCALE, MAX_SCALE);
        const k = s2 / v.s;
        const ax = prev.x - rect.left - rect.width / 2;
        const ay = prev.y - rect.top - rect.height / 2;
        commit(
          clampPan({
            s: s2,
            // Zoom about the previous midpoint, then follow the midpoint itself
            // so a two-finger drag also pans.
            x: ax - k * (ax - v.x) + (mid.x - prev.x),
            y: ay - k * (ay - v.y) + (mid.y - prev.y),
          }),
        );
        return;
      }

      const drag = dragRef.current;
      if (!drag) return;
      const v = viewRef.current;
      commit(
        clampPan({
          s: v.s,
          x: v.x + (e.clientX - drag.x),
          y: v.y + (e.clientY - drag.y),
        }),
      );
      dragRef.current = { x: e.clientX, y: e.clientY };
    },
    [clampPan, commit],
  );

  const endPointer = useCallback((e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) pinchRef.current = null;
    const rest = [...pointers.current.values()][0];
    dragRef.current = rest ? { x: rest.x, y: rest.y } : null;
  }, []);

  // Double click / double tap toggles between fit and 2× fit.
  const onDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const fit = computeFit();
      const zoomedIn = viewRef.current.s > fit * 1.05;
      if (zoomedIn) {
        commit({ s: fit, x: 0, y: 0 });
      } else {
        zoomAt(2, e.clientX, e.clientY);
      }
    },
    [commit, computeFit, zoomAt],
  );

  /** Centre of the viewport, used to anchor button-driven zoom. */
  const viewportCenter = useCallback((): [number, number] => {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect) return [0, 0];
    return [rect.left + rect.width / 2, rect.top + rect.height / 2];
  }, []);

  const percent = Math.round(view.s * 100);
  const stageStyle = size
    ? { width: size.w, height: size.h, transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }
    : { transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` };

  return (
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={title || t.diagramZoom}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="lightbox-box" ref={boxRef} tabIndex={-1}>
        <div className="lightbox-head">
          <span className="lightbox-title">{title || t.diagram}</span>
          <span className="lightbox-hint">{t.zoomHint}</span>
          <div className="lightbox-tools">
            <button
              type="button"
              className="lightbox-tool lightbox-icon"
              onClick={() => zoomAt(1 / 1.3, ...viewportCenter())}
              aria-label={t.zoomOut}
              title={t.zoomOut}
            >
              −
            </button>
            <button
              type="button"
              className="lightbox-tool lightbox-fit"
              onClick={applyFit}
              aria-label={t.zoomFit}
              title={t.zoomFit}
            >
              <span className="lightbox-fit-icon" aria-hidden="true">
                ⤢
              </span>
              <span className="lightbox-percent">{percent}%</span>
            </button>
            <button
              type="button"
              className="lightbox-tool lightbox-icon"
              onClick={() => zoomAt(1.3, ...viewportCenter())}
              aria-label={t.zoomIn}
              title={t.zoomIn}
            >
              +
            </button>
            <button
              type="button"
              className="lightbox-tool lightbox-icon lightbox-close"
              onClick={onClose}
              aria-label={t.zoomClose}
              title={t.zoomClose}
            >
              ✕
            </button>
          </div>
        </div>
        <div
          className="lightbox-viewport"
          ref={viewportRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endPointer}
          onPointerCancel={endPointer}
          onDoubleClick={onDoubleClick}
        >
          <div
            className="lightbox-stage"
            ref={stageRef}
            style={stageStyle}
            // Re-hosts only the site's own mermaid SVG output.
            dangerouslySetInnerHTML={{ __html: svgHtml }}
          />
        </div>
      </div>
    </div>
  );
}
