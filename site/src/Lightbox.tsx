import { useCallback, useEffect, useRef, useState } from "react";

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}

/**
 * Click-to-zoom lightbox for document diagrams. The diagram SVG is passed in
 * as a string and re-hosted inside a pannable/zoomable viewport. Wheel zooms,
 * pointer-drag pans, buttons set scale, Esc / overlay click / button close.
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
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const dragging = useRef(false);
  const last = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  const reset = useCallback(() => {
    setScale(1);
    setTx(0);
    setTy(0);
  }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setScale((s) => clamp(s * (e.deltaY < 0 ? 1.15 : 0.87), 0.25, 4));
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    dragging.current = true;
    last.current = { x: e.clientX, y: e.clientY };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    const dx = e.clientX - last.current.x;
    const dy = e.clientY - last.current.y;
    last.current = { x: e.clientX, y: e.clientY };
    setTx((x) => x + dx);
    setTy((y) => y + dy);
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    dragging.current = false;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
  }, []);

  return (
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={title || "图表放大"}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape") onClose();
      }}
    >
      <div className="lightbox-box">
        <div className="lightbox-head">
          <span className="lightbox-title">{title || "图表"}</span>
          <span className="lightbox-hint">滚轮缩放 · 拖拽平移 · Esc 关闭</span>
          <div className="lightbox-tools">
            <button
              type="button"
              className="lightbox-tool"
              onClick={() => setScale((s) => clamp(s / 1.25, 0.25, 4))}
              aria-label="缩小"
            >
              −
            </button>
            <button
              type="button"
              className="lightbox-tool"
              onClick={() => setScale((s) => clamp(s * 1.25, 0.25, 4))}
              aria-label="放大"
            >
              +
            </button>
            <button type="button" className="lightbox-tool" onClick={reset}>
              100%
            </button>
            <button
              type="button"
              className="lightbox-tool lightbox-close"
              onClick={onClose}
              aria-label="关闭"
            >
              ✕
            </button>
          </div>
        </div>
        <div
          className="lightbox-viewport"
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <div
            className="lightbox-stage"
            style={{
              transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
            }}
            // Re-hosts only the site's own mermaid SVG output.
            dangerouslySetInnerHTML={{ __html: svgHtml }}
          />
        </div>
      </div>
    </div>
  );
}
