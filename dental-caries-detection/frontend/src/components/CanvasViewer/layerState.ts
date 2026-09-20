import { useCallback, useState } from 'react';

export interface LayerVisibility {
  boxes: boolean;
  masks: boolean;
  axes: boolean;
  labels: boolean;
}

export const defaultLayerVisibility: LayerVisibility = {
  boxes: true,
  masks: true,
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
    setLayers({ boxes: visible, masks: visible, axes: visible, labels: visible });
  }, []);

  return { layers, toggle, setAll };
}
