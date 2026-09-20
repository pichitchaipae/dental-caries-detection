import { useCallback, useEffect, useLayoutEffect, useRef, type RefObject } from 'react';
import type { ToothViewModel } from '../../features/analysis/analysisTypes';
import type { LayerVisibility } from './layerState';
import { drawAxes, drawBoundingBox, drawLabel, drawMask, drawSelectionHalo } from './overlays';
import { decodeMask } from '../../lib/rle';

interface Viewport {
  scale: number;
  offsetX: number;
  offsetY: number;
}

interface ImageFilter {
  brightness: number; // percent, 100 = unchanged
  contrast: number; // percent, 100 = unchanged
}

interface UseCanvasRendererArgs {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  image: HTMLImageElement | null;
  imageSize: { width: number; height: number };
  teeth: ToothViewModel[];
  selectedToothId: number | null;
  layers: LayerVisibility;
  imageFilter: ImageFilter;
  onSelectTooth: (id: number | null) => void;
}

interface UseCanvasRendererResult {
  fitToView: () => void;
  actualSize: () => void;
}

const MIN_SCALE = 0.2;
const MAX_SCALE = 8;
const ZOOM_STEP = 1.1;
const CLICK_VS_DRAG_THRESHOLD_PX = 4;

function computeFitViewport(
  canvas: HTMLCanvasElement,
  imageSize: { width: number; height: number }
): Viewport {
  const fitScale = Math.min(canvas.width / imageSize.width, canvas.height / imageSize.height);
  return {
    scale: fitScale,
    offsetX: (canvas.width - imageSize.width * fitScale) / 2,
    offsetY: (canvas.height - imageSize.height * fitScale) / 2,
  };
}

// FE-6.1/6.3: draw loop (rAF-coalesced), pan (drag) / zoom (wheel) /
// click-to-select, DPR-aware canvas sizing, and Fit / 1:1 view controls.
export function useCanvasRenderer({
  canvasRef,
  image,
  imageSize,
  teeth,
  selectedToothId,
  layers,
  imageFilter,
  onSelectTooth,
}: UseCanvasRendererArgs): UseCanvasRendererResult {
  const viewportRef = useRef<Viewport>({ scale: 1, offsetX: 0, offsetY: 0 });
  const draggingRef = useRef<{ x: number; y: number } | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const drawRef = useRef<() => void>(() => {});

  // rAF-coalesced redraw: any number of calls within one frame collapse into
  // a single draw (FE-6.3: "coalesces multiple requestRedraw calls into one
  // frame"). `drawRef.current` always holds the latest closure (updated in
  // the effect below), so this function itself never needs to change identity.
  const requestRedraw = useCallback(() => {
    if (rafIdRef.current !== null) return;
    rafIdRef.current = requestAnimationFrame(() => {
      rafIdRef.current = null;
      drawRef.current();
    });
  }, []);

  useEffect(() => {
    return () => {
      // Must also null out the ref, not just cancel: React 18 StrictMode
      // double-invokes effects in dev (mount -> cleanup -> mount again on the
      // same instance, refs intact), so this cleanup fires once before any
      // real unmount. Leaving a stale non-null id here would make every
      // future requestRedraw() believe a frame is still pending and skip
      // scheduling forever, permanently freezing the canvas.
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, []);

  // DPR-aware, container-responsive backing store. Runs before paint
  // (useLayoutEffect) so there's no visible flash at the default 300x150
  // canvas size. The pan/zoom coordinate math elsewhere is DPR-agnostic
  // already (it works in canvas.width/height ratios), so no other code
  // needs to know about devicePixelRatio.
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const container = canvas?.parentElement;
    if (!canvas || !container) return;

    const applySize = () => {
      const dpr = window.devicePixelRatio || 1;
      const cssWidth = container.clientWidth;
      const cssHeight = container.clientHeight;
      if (cssWidth === 0 || cssHeight === 0) return;

      const nextWidth = Math.max(1, Math.round(cssWidth * dpr));
      const nextHeight = Math.max(1, Math.round(cssHeight * dpr));
      const sizeChanged = canvas.width !== nextWidth || canvas.height !== nextHeight;
      if (sizeChanged) {
        canvas.width = nextWidth;
        canvas.height = nextHeight;
      }

      if (sizeChanged && imageSize.width > 0 && imageSize.height > 0) {
        viewportRef.current = computeFitViewport(canvas, imageSize);
      }
      requestRedraw();
    };

    applySize();
    const observer = new ResizeObserver(applySize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [canvasRef, imageSize, requestRedraw]);

  const fitToView = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || imageSize.width === 0 || imageSize.height === 0) return;
    viewportRef.current = computeFitViewport(canvas, imageSize);
    requestRedraw();
  }, [canvasRef, imageSize, requestRedraw]);

  const actualSize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || imageSize.width === 0 || imageSize.height === 0) return;
    viewportRef.current = {
      scale: 1,
      offsetX: (canvas.width - imageSize.width) / 2,
      offsetY: (canvas.height - imageSize.height) / 2,
    };
    requestRedraw();
  }, [canvasRef, imageSize.width, imageSize.height, requestRedraw]);

  // Keep the draw closure current on every render, but only *actually* draw
  // when something visually relevant changes (the effect below) or a pointer
  // interaction calls requestRedraw() directly.
  useEffect(() => {
    drawRef.current = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;

      const { scale, offsetX, offsetY } = viewportRef.current;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(offsetX, offsetY);
      ctx.scale(scale, scale);

      if (image) {
        // Filter applies only to the source image, never to overlays: reset
        // before drawing overlays so caries-red / sound-blue fills stay
        // visually consistent regardless of brightness/contrast.
        ctx.filter = `brightness(${imageFilter.brightness}%) contrast(${imageFilter.contrast}%)`;
        ctx.drawImage(image, 0, 0, imageSize.width, imageSize.height);
        ctx.filter = 'none';
      }

      // Layered draw loop per project-structure.md 10.1: boxes, then masks,
      // then axes, then labels — one pass per layer, not per tooth — with
      // the selected tooth's halo drawn last, on top of everything.
      if (layers.boxes) {
        for (const tooth of teeth) drawBoundingBox(ctx, tooth);
      }
      if (layers.masks) {
        for (const tooth of teeth) drawMask(ctx, tooth, imageSize.width, imageSize.height);
      }
      if (layers.axes) {
        for (const tooth of teeth) drawAxes(ctx, tooth);
      }
      if (layers.labels) {
        for (const tooth of teeth) drawLabel(ctx, tooth);
      }
      if (selectedToothId !== null) {
        const selected = teeth.find((tooth) => tooth.id === selectedToothId);
        if (selected) drawSelectionHalo(ctx, selected);
      }

      ctx.restore();
    };
  });

  useEffect(() => {
    requestRedraw();
  }, [image, teeth, selectedToothId, layers, imageFilter, imageSize, requestRedraw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const toImageCoords = (clientX: number, clientY: number): [number, number] => {
      const rect = canvas.getBoundingClientRect();
      const { scale, offsetX, offsetY } = viewportRef.current;
      const canvasX = ((clientX - rect.left) / rect.width) * canvas.width;
      const canvasY = ((clientY - rect.top) / rect.height) * canvas.height;
      return [(canvasX - offsetX) / scale, (canvasY - offsetY) / scale];
    };

    // FE-6.3: test against each tooth's mask path via ctx.isPointInPath
    // first (topmost tooth first), falling back to its bbox if the click is
    // inside the box but misses the (possibly small/irregular) mask polygon.
    const hitTest = (imgX: number, imgY: number): ToothViewModel | null => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;

      ctx.save();
      // isPointInPath compares the point against the path in whatever
      // transform is active *now*; both the mask/bbox coordinates and
      // imgX/imgY are already in untransformed image space, so reset to
      // identity rather than double-applying the pan/zoom transform.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      try {
        for (let i = teeth.length - 1; i >= 0; i -= 1) {
          const tooth = teeth[i];
          const rings = decodeMask(tooth.mask, imageSize.width, imageSize.height);
          const maskPath = new Path2D();
          for (const ring of rings) {
            if (ring.length === 0) continue;
            maskPath.moveTo(ring[0][0], ring[0][1]);
            for (const [px, py] of ring.slice(1)) maskPath.lineTo(px, py);
            maskPath.closePath();
          }
          if (ctx.isPointInPath(maskPath, imgX, imgY)) return tooth;
        }

        for (let i = teeth.length - 1; i >= 0; i -= 1) {
          const tooth = teeth[i];
          const [x, y, w, h] = tooth.bbox;
          const bboxPath = new Path2D();
          bboxPath.rect(x, y, w, h);
          if (ctx.isPointInPath(bboxPath, imgX, imgY)) return tooth;
        }

        return null;
      } finally {
        ctx.restore();
      }
    };

    const handlePointerDown = (event: PointerEvent) => {
      draggingRef.current = { x: event.clientX, y: event.clientY };
    };

    const handlePointerMove = (event: PointerEvent) => {
      const drag = draggingRef.current;
      if (!drag) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      draggingRef.current = { x: event.clientX, y: event.clientY };
      viewportRef.current = {
        ...viewportRef.current,
        offsetX: viewportRef.current.offsetX + dx,
        offsetY: viewportRef.current.offsetY + dy,
      };
      requestRedraw();
    };

    const handlePointerUp = (event: PointerEvent) => {
      const drag = draggingRef.current;
      draggingRef.current = null;
      if (!drag) return;

      const dist = Math.hypot(event.clientX - drag.x, event.clientY - drag.y);
      if (dist > CLICK_VS_DRAG_THRESHOLD_PX) return; // it was a pan, not a click

      const [imgX, imgY] = toImageCoords(event.clientX, event.clientY);
      const hit = hitTest(imgX, imgY);
      onSelectTooth(hit ? hit.id : null);
    };

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const pivotX = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const pivotY = ((event.clientY - rect.top) / rect.height) * canvas.height;
      const { scale, offsetX, offsetY } = viewportRef.current;
      const zoomFactor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
      const newScale = Math.min(Math.max(scale * zoomFactor, MIN_SCALE), MAX_SCALE);

      // Keep the point under the cursor fixed while zooming.
      viewportRef.current = {
        scale: newScale,
        offsetX: pivotX - ((pivotX - offsetX) / scale) * newScale,
        offsetY: pivotY - ((pivotY - offsetY) / scale) * newScale,
      };
      requestRedraw();
    };

    canvas.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    canvas.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      canvas.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [canvasRef, teeth, imageSize.width, imageSize.height, onSelectTooth, requestRedraw]);

  return { fitToView, actualSize };
}
