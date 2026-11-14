import { z } from 'zod';

export type ProcessStatus = 'idle' | 'processing' | 'done' | 'fail';

export interface SurfaceFinding {
  name: 'mesial' | 'distal' | 'occlusal';
  label: 'caries';
  probability: number | null;
  method: 'Geometry_Heuristic' | 'RF' | 'XThirds_Fallback' | 'CNN';
}

export type SurfaceName = SurfaceFinding['name'];

// One predicted lesion in pano coordinates (API v2 draft, docs-md/
// api-contract-v2-draft.md §2.1). `surface` is null until ML sends it
// (ML-10). `probability` is the caries detector score, not calibrated.
// The v1.1 `bbox` / `confidence` fields no longer exist on the wire.
export interface CariesLesion {
  id: number; // unique within its tooth (0..n-1)
  surface: SurfaceName | null;
  probability: number;
  mask: { encoding: 'polygon'; data: [number, number][] };
  centroid: [number, number];
}

export interface ToothAxes {
  major: [number, number];
  minor: [number, number];
  rotation_deg: number;
  clamped: boolean;
}

export interface MaskData {
  encoding: 'polygon' | 'rle';
  data: [number, number][] | number[];
}

export interface Tooth {
  id: number;
  fdi: number;
  bbox: [number, number, number, number]; // tooth bbox only: label anchor + hit-test fallback, never drawn
  mask: MaskData;
  axes: ToothAxes;
  has_caries: boolean;
  surfaces: SurfaceFinding[];
  caries: CariesLesion[]; // required in v2; [] for a tooth without lesions
}

export interface InferenceData {
  meta: {
    job_id: number;
    processed_at: string;
    models: { detector: string; classifier: string };
    timings_ms: Record<string, number>;
  };
  image: { width: number; height: number };
  teeth: Tooth[];
}

const pointSchema = z.tuple([z.number(), z.number()]);
const surfaceFindingSchema = z.object({
  name: z.enum(['mesial', 'distal', 'occlusal']),
  label: z.literal('caries'),
  probability: z.number().min(0).max(1).nullable(),
  method: z.enum(['Geometry_Heuristic', 'RF', 'XThirds_Fallback', 'CNN']),
});
const cariesLesionSchema = z.object({
  id: z.number().int().nonnegative(),
  surface: z.enum(['mesial', 'distal', 'occlusal']).nullable(),
  probability: z.number().min(0).max(1),
  // No `.min(3)`: the backend adapter does not validate lesion polygons, and
  // one degenerate lesion must not fail the whole result. Rendering skips
  // polygons with fewer than 3 points (overlays.drawCariesMasks).
  mask: z.object({ encoding: z.literal('polygon'), data: z.array(pointSchema) }),
  centroid: pointSchema,
});
// z.object strips unknown keys, so a stray v1 `teeth[].confidence` is dropped
// at parse time rather than leaking into the view model.
const toothSchema = z.object({
  id: z.number().int(),
  fdi: z.number().int().min(11).max(48),
  bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  mask: z.discriminatedUnion('encoding', [
    z.object({ encoding: z.literal('polygon'), data: z.array(pointSchema).min(3) }),
    z.object({ encoding: z.literal('rle'), data: z.array(z.number()) }),
  ]),
  axes: z.object({
    major: pointSchema,
    minor: pointSchema,
    rotation_deg: z.number(),
    clamped: z.boolean(),
  }),
  has_caries: z.boolean(),
  surfaces: z.array(surfaceFindingSchema).max(3),
  caries: z.array(cariesLesionSchema),
});
const inferenceDataSchema = z.object({
  meta: z.object({
    job_id: z.number().int(),
    processed_at: z.string(),
    models: z.object({ detector: z.string(), classifier: z.string() }),
    timings_ms: z.record(z.string(), z.number()),
  }),
  image: z.object({
    width: z.number().int().positive(),
    height: z.number().int().positive(),
  }),
  teeth: z.array(toothSchema),
});

const processResponseSchema = z.discriminatedUnion('status', [
  z.object({ status: z.literal('idle') }),
  z.object({ status: z.literal('processing') }),
  z.object({ status: z.literal('fail'), fail_message: z.string() }),
  z.object({ status: z.literal('done'), image_base64: z.string(), data: inferenceDataSchema }),
]);

export type ProcessResponse = z.infer<typeof processResponseSchema>;

export function parseProcessResponse(json: unknown): ProcessResponse {
  return processResponseSchema.parse(json);
}
