import { useEffect, useRef, useState } from 'react';
import type { ToothViewModel } from '../../features/analysis/analysisTypes';
import { useCanvasRenderer } from './useCanvasRenderer';
import { useLayerState, type LayerKey } from './layerState';

interface CanvasViewerProps {
  imageBase64: string;
  imageSize: { width: number; height: number };
  teeth: ToothViewModel[];
  selectedToothId: number | null;
  onSelectTooth: (id: number | null) => void;
}

const LAYER_LABELS: Record<LayerKey, string> = {
  boxes: 'Boxes',
  masks: 'Masks',
  axes: 'Axes',
  labels: 'Labels',
};

export function CanvasViewer({
  imageBase64,
  imageSize,
  teeth,
  selectedToothId,
  onSelectTooth,
}: CanvasViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const { layers, toggle } = useLayerState();
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);

  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.src = imageBase64;
    // FE-7.2: decode off the main thread before first paint, so a large
    // base64 OPG doesn't stall on first `drawImage` inside the render loop.
    img
      .decode()
      .catch(() => {
        // Some environments resolve `decode()` before `onload`/never resolve
        // it for data URIs; fall back to whatever the image ends up with.
      })
      .finally(() => {
        if (!cancelled) setImage(img);
      });
    return () => {
      cancelled = true;
      setImage(null);
    };
  }, [imageBase64]);

  const { fitToView, actualSize } = useCanvasRenderer({
    canvasRef,
    image,
    imageSize,
    teeth,
    selectedToothId,
    layers,
    imageFilter: { brightness, contrast },
    onSelectTooth,
  });

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-slate-600">
          {(Object.keys(layers) as LayerKey[]).map((key) => (
            <label key={key} className="flex cursor-pointer items-center gap-1.5">
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={() => toggle(key)}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              {LAYER_LABELS[key]}
            </label>
          ))}
        </div>
        <div className="flex items-center gap-2 text-sm">
          <button
            type="button"
            onClick={fitToView}
            className="rounded-md border border-slate-300 px-2.5 py-1 font-medium text-slate-600 transition hover:bg-slate-50"
          >
            Fit
          </button>
          <button
            type="button"
            onClick={actualSize}
            className="rounded-md border border-slate-300 px-2.5 py-1 font-medium text-slate-600 transition hover:bg-slate-50"
          >
            1:1
          </button>
        </div>
      </div>

      {/* This wrapper defines the visible box; the canvas fills it exactly
          and useCanvasRenderer sizes its backing store to match
          (container CSS size x devicePixelRatio) via ResizeObserver, so the
          image stays crisp on HiDPI screens instead of being CSS-upscaled
          from a fixed low-res backing store. */}
      <div className="aspect-video w-full overflow-hidden rounded-lg bg-slate-950">
        <canvas
          ref={canvasRef}
          className="h-full w-full cursor-grab touch-none active:cursor-grabbing"
        />
      </div>

      <div className="flex flex-col gap-2 text-sm text-slate-600">
        <label className="flex items-center gap-3">
          <span className="w-20 shrink-0">Brightness</span>
          <input
            type="range"
            min={50}
            max={150}
            value={brightness}
            onChange={(event) => setBrightness(Number(event.target.value))}
            className="h-1.5 flex-1 accent-brand-600"
          />
          <span className="w-10 shrink-0 text-right tabular-nums">{brightness}%</span>
        </label>
        <label className="flex items-center gap-3">
          <span className="w-20 shrink-0">Contrast</span>
          <input
            type="range"
            min={50}
            max={150}
            value={contrast}
            onChange={(event) => setContrast(Number(event.target.value))}
            className="h-1.5 flex-1 accent-brand-600"
          />
          <span className="w-10 shrink-0 text-right tabular-nums">{contrast}%</span>
        </label>
      </div>

      {teeth.length === 0 ? (
        <p className="text-xs font-medium text-amber-600">
          No teeth detected in this image. Scroll to zoom, drag to pan.
        </p>
      ) : (
        <p className="text-xs text-slate-400">
          Scroll to zoom, drag to pan, click a tooth to select it.
        </p>
      )}
    </div>
  );
}
