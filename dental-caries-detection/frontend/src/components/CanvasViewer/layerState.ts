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
