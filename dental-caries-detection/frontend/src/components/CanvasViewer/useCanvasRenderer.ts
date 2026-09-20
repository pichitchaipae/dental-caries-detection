import { useEffect, useRef, useState, type RefObject } from 'react';
import type { Tooth } from '../../domain/inference';
import type { LayerVisibility } from './layerState';
import { drawAxes, drawBoundingBox, drawLabel, drawMask } from './overlays';
import { decodeMask, isPointInPolygon } from '../../lib/rle';

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
  teeth: Tooth[];
  selectedToothId: number | null;
  layers: LayerVisibility;
  imageFilter: ImageFilter;
  onSelectTooth: (id: number | null) => void;
}

const MIN_SCALE = 0.2;
const MAX_SCALE = 8;
const ZOOM_STEP = 1.1;
const CLICK_VS_DRAG_THRESHOLD_PX = 4;

// Draw loop, plus pan (drag) / zoom (wheel) / click-to-select. The viewport
// lives in a ref (not state) because it changes on every pointer-move frame;
// `bumpRedraw` is the one piece of state used to trigger a re-render + redraw
// after a ref-only viewport change.
export function useCanvasRenderer({
  canvasRef,
  image,
  imageSize,
  teeth,
  selectedToothId,
  layers,
  imageFilter,
  onSelectTooth,
}: UseCanvasRendererArgs): void {
  const viewportRef = useRef<Viewport>({ scale: 1, offsetX: 0, offsetY: 0 });
  const draggingRef = useRef<{ x: number; y: number } | null>(null);
  const [, bumpRedraw] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || imageSize.width === 0 || imageSize.height === 0) return;
    const fitScale = Math.min(canvas.width / imageSize.width, canvas.height / imageSize.height);
    viewportRef.current = {
      scale: fitScale,
      offsetX: (canvas.width - imageSize.width * fitScale) / 2,
      offsetY: (canvas.height - imageSize.height * fitScale) / 2,
    };
    bumpRedraw((n) => n + 1);
  }, [canvasRef, imageSize.width, imageSize.height]);

  useEffect(() => {
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
      // before drawing boxes/masks/axes/labels so caries-red / sound-blue
      // fills stay visually consistent regardless of the brightness/contrast
      // the clinician has dialed in.
      ctx.filter = `brightness(${imageFilter.brightness}%) contrast(${imageFilter.contrast}%)`;
      ctx.drawImage(image, 0, 0, imageSize.width, imageSize.height);
      ctx.filter = 'none';
    }

    for (const tooth of teeth) {
      const isSelected = tooth.id === selectedToothId;
      if (layers.masks) drawMask(ctx, tooth, imageSize.width, imageSize.height);
      if (layers.boxes) drawBoundingBox(ctx, tooth, isSelected);
      if (layers.axes) drawAxes(ctx, tooth);
      if (layers.labels) drawLabel(ctx, tooth);
    }

    ctx.restore();
    // Intentionally no dependency array: re-runs after every render (including
    // the bumpRedraw-triggered ones from pan/zoom, which only touch the ref).
  });

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

    const hitTest = (imgX: number, imgY: number): Tooth | null => {
      for (let i = teeth.length - 1; i >= 0; i -= 1) {
        const tooth = teeth[i];
        const rings = decodeMask(tooth.mask, imageSize.width, imageSize.height);
        if (rings.some((ring) => isPointInPolygon([imgX, imgY], ring))) return tooth;
      }
      return null;
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
      bumpRedraw((n) => n + 1);
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
      bumpRedraw((n) => n + 1);
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
  }, [canvasRef, teeth, imageSize.width, imageSize.height, onSelectTooth]);
}
