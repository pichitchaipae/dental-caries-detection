import { z } from 'zod';

// Mirrors what the backend actually sends on GET /process: the ml-service result
// JSON forwarded untouched (see backend/src/routes/process.ts and
// backend/src/fixtures/ml-result.raw.sample.json). The backend cannot be changed,
// so the wire shape is parsed as-is (`RawInferenceData`) and then adapted into the
// smaller `InferenceData` the UI consumes (`normalizeInference`).

export type ProcessStatus = 'idle' | 'processing' | 'done' | 'fail';

export interface SurfaceFinding {
  name: 'mesial' | 'distal' | 'occlusal' | 'buccal' | 'lingual';
  label: 'caries' | 'sound';
  probability: number;
  method?: string;
}

export interface ToothAxes {
  major: [number, number];
  minor: [number, number];
  rotation_deg: number;
  clamped?: boolean;
}

export interface MaskData {
  encoding: 'polygon' | 'rle';
  // polygon: [[x, y], ...]; rle: flat run lengths (not emitted in Phase 1).
  data: [number, number][] | number[];
}

export interface Tooth {
  id: number;
  fdi: number;
  confidence: number;
  bbox: [number, number, number, number]; // x, y, w, h
  mask: MaskData;
  axes: ToothAxes;
  surfaces: SurfaceFinding[];
}

// What the UI consumes. `image` is derived from the wire `meta.image_size`; the
// rest of the wire `meta` (job_id, completed_at, counts) is not used by any UI.
export interface InferenceData {
  image: { width: number; height: number };
  teeth: Tooth[];
}

export type ProcessResponse =
  | { status: 'idle' }
  | { status: 'processing' }
  | { status: 'fail'; fail_message: string }
  | { status: 'done'; image_base64: string; data: InferenceData };

// --- zod schema: the executable contract. This is the single source of truth
// on the frontend side for what a response must look like; `processClient.ts`
// never trusts a response it hasn't been parsed through. ---

const surfaceFindingSchema = z.object({
  name: z.enum(['mesial', 'distal', 'occlusal', 'buccal', 'lingual']),
  label: z.enum(['caries', 'sound']),
  probability: z.number().min(0).max(1),
  method: z.string().optional(),
});

const toothAxesSchema = z.object({
  major: z.tuple([z.number(), z.number()]),
  minor: z.tuple([z.number(), z.number()]),
  rotation_deg: z.number(),
  clamped: z.boolean().optional(),
});

const maskDataSchema = z.object({
  encoding: z.enum(['polygon', 'rle']),
  data: z.union([z.array(z.tuple([z.number(), z.number()])), z.array(z.number())]),
});

const toothSchema = z.object({
  id: z.number(),
  fdi: z.number(),
  confidence: z.number().min(0).max(1),
  bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  mask: maskDataSchema,
  axes: toothAxesSchema,
  surfaces: z.array(surfaceFindingSchema),
});

// Only `image_size` is required (the UI needs it); the other meta fields are
// accepted but optional so a missing counter never blocks rendering results.
const rawInferenceMetaSchema = z.object({
  job_id: z.number().optional(),
  completed_at: z.string().optional(),
  image_size: z.object({ width: z.number(), height: z.number() }),
  tooth_count: z.number().optional(),
  caries_count: z.number().optional(),
});

const rawInferenceDataSchema = z.object({
  meta: rawInferenceMetaSchema,
  teeth: z.array(toothSchema),
});

export type RawInferenceData = z.infer<typeof rawInferenceDataSchema>;

export function normalizeInference(raw: RawInferenceData): InferenceData {
  return { image: raw.meta.image_size, teeth: raw.teeth };
}

const inferenceDataSchema = rawInferenceDataSchema.transform(normalizeInference);

const processResponseSchema = z.discriminatedUnion('status', [
  z.object({ status: z.literal('idle') }),
  z.object({ status: z.literal('processing') }),
  z.object({ status: z.literal('fail'), fail_message: z.string() }),
  z.object({ status: z.literal('done'), image_base64: z.string(), data: inferenceDataSchema }),
]);

export function parseProcessResponse(json: unknown): ProcessResponse {
  return processResponseSchema.parse(json);
}
