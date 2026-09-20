import type { InferenceData, MaskData, SurfaceFinding, ToothAxes } from '../../domain/inference';

// `file`/`previewUrl` for the selected image live in AnalysisView's separate
// `preview` state (not here), since the left "Input Workspace" panel needs to
// keep showing the thumbnail through 'processing'/'done'/'fail' too, not just
// while the phase is still 'selected'.
export type WorkflowState =
  | { phase: 'empty' }
  | { phase: 'selected'; file: File }
  | { phase: 'submitting'; file: File }
  | { phase: 'processing' }
  | { phase: 'done'; imageBase64: string; data: InferenceData }
  | { phase: 'fail'; failMessage: string };

// FE-5.4: adapts the wire-format InferenceData into what the viewer/panels
// actually consume — precomputed display strings and a stable per-tooth
// identity color, so CanvasViewer/ToothDetailPanel/FindingsTable don't each
// recompute the same derived values. Pure, no React imports.

// Colorblind-considerate, mutually distinct from the semantic mask colors
// (caries-red / sound-blue in overlays.ts) and the orange selection halo —
// this palette identifies *which* tooth, not its clinical status.
const IDENTITY_COLOR_PALETTE = [
  '#2563eb', // blue
  '#7c3aed', // violet
  '#0891b2', // cyan
  '#059669', // emerald
  '#db2777', // pink
  '#4f46e5', // indigo
] as const;

export interface ToothViewModel {
  id: number;
  fdi: number;
  displayLabel: string; // "FDI 36"
  confidence: number;
  bbox: [number, number, number, number];
  mask: MaskData;
  axes: ToothAxes;
  surfaces: SurfaceFinding[];
  cariesCount: number;
  cariesSummary: string; // "2 / 5 surfaces"
  hasCaries: boolean;
  colorKey: string; // stable per-tooth identity color, for overlays
}

export interface AnalysisSummary {
  totalTeeth: number;
  teethWithCaries: number;
  totalCariesSurfaces: number;
}

export interface AnalysisViewModel {
  image: { width: number; height: number };
  teeth: ToothViewModel[];
  summary: AnalysisSummary;
}

function toToothViewModel(tooth: InferenceData['teeth'][number]): ToothViewModel {
  const cariesCount = tooth.surfaces.filter((surface) => surface.label === 'caries').length;

  return {
    id: tooth.id,
    fdi: tooth.fdi,
    displayLabel: `FDI ${tooth.fdi}`,
    confidence: tooth.confidence,
    bbox: tooth.bbox,
    mask: tooth.mask,
    axes: tooth.axes,
    surfaces: tooth.surfaces,
    cariesCount,
    cariesSummary: `${cariesCount} / ${tooth.surfaces.length} surfaces`,
    hasCaries: cariesCount > 0,
    colorKey: IDENTITY_COLOR_PALETTE[tooth.id % IDENTITY_COLOR_PALETTE.length],
  };
}

export function toViewModel(data: InferenceData): AnalysisViewModel {
  const teeth = data.teeth.map(toToothViewModel);

  return {
    image: data.image,
    teeth,
    summary: {
      totalTeeth: teeth.length,
      teethWithCaries: teeth.filter((tooth) => tooth.hasCaries).length,
      totalCariesSurfaces: teeth.reduce((sum, tooth) => sum + tooth.cariesCount, 0),
    },
  };
}
