import { useCallback, useState } from 'react';

// `masks` = whole-tooth masks; `caries` = per-lesion masks (API v2). The old
// `boxes` layer is gone — bounding boxes are no longer drawn.
export interface LayerVisibility {
  masks: boolean;
  caries: boolean;
  axes: boolean;
  labels: boolean;
}

export const defaultLayerVisibility: LayerVisibility = {
  masks: true,
  caries: true,
  axes: false,
  labels: true,
};

export type LayerKey = keyof LayerVisibility;

// FE-6.4: layer-toggle state as its own reusable hook, consumed read-only by
// Naris's useCanvasRenderer and read/write by CanvasViewer's toggle control bar.
export function useLayerState(): {
  layers: LayerVisibility;
  toggle: (key: LayerKey) => void;
  setAll: (visible: boolean) => void;
} {
  const [layers, setLayers] = useState<LayerVisibility>(defaultLayerVisibility);

  const toggle = useCallback((key: LayerKey) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const setAll = useCallback((visible: boolean) => {
    setLayers({ masks: visible, caries: visible, axes: visible, labels: visible });
  }, []);

  return { layers, toggle, setAll };
}
