import { z } from 'zod';

// Mirrors docs-md/project-structure.md Section 7.1 exactly. Kept byte-aligned
// with the backend's route schemas (INT-1) and backend/src/fixtures/result.sample.json
// (INT-2) once that fixture exists.

export type ProcessStatus = 'idle' | 'processing' | 'done' | 'fail';

export interface SurfaceFinding {
  name: 'mesial' | 'distal' | 'occlusal' | 'buccal' | 'lingual';
  label: 'caries' | 'sound';
  probability: number;
}

export interface ToothAxes {
  major: [number, number];
  minor: [number, number];
  rotation_deg: number;
}

export interface MaskData {
  encoding: 'polygon' | 'rle';
  // polygon: [[x, y], ...]; rle: flat run lengths (not emitted in Phase 1).
  data: number[][] | number[];
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

export interface InferenceMeta {
  processed_at: string;
  models: { detector: string; classifier: string };
  timings_ms: Record<string, number>;
}

export interface InferenceData {
  meta: InferenceMeta;
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
});

const toothAxesSchema = z.object({
  major: z.tuple([z.number(), z.number()]),
  minor: z.tuple([z.number(), z.number()]),
  rotation_deg: z.number(),
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

const inferenceMetaSchema = z.object({
  processed_at: z.string(),
  models: z.object({ detector: z.string(), classifier: z.string() }),
  timings_ms: z.record(z.string(), z.number()),
});

const inferenceDataSchema = z.object({
  meta: inferenceMetaSchema,
  image: z.object({ width: z.number(), height: z.number() }),
  teeth: z.array(toothSchema),
});

const processResponseSchema = z.discriminatedUnion('status', [
  z.object({ status: z.literal('idle') }),
  z.object({ status: z.literal('processing') }),
  z.object({ status: z.literal('fail'), fail_message: z.string() }),
  z.object({ status: z.literal('done'), image_base64: z.string(), data: inferenceDataSchema }),
]);

export function parseProcessResponse(json: unknown): ProcessResponse {
  return processResponseSchema.parse(json);
}
