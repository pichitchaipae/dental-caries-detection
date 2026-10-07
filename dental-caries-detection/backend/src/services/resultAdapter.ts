import { z } from 'zod';

const pointSchema = z.tuple([z.number(), z.number()]);
const rawSurfaceSchema = z.object({
  name: z.string(),
  label: z.string().optional(),
  probability: z.number().min(0).max(1).optional(),
  method: z.string().optional(),
});
const rawToothSchema = z.object({
  id: z.number().int().optional(),
  fdi: z.number().int().min(11).max(48),
  confidence: z.number().min(0).max(1),
  bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  mask: z.object({
    encoding: z.literal('polygon'),
    data: z.array(pointSchema),
  }),
  axes: z
    .object({
      major: pointSchema,
      minor: pointSchema,
      rotation_deg: z.number(),
      clamped: z.boolean().optional(),
    })
    .optional(),
  surfaces: z.array(rawSurfaceSchema),
});
const rawResultSchema = z.object({
  meta: z.object({
    job_id: z.number().int(),
    completed_at: z.string(),
    image_size: z.object({
      width: z.number().int().positive(),
      height: z.number().int().positive(),
    }),
    models: z
      .object({ detector: z.string(), classifier: z.string() })
      .optional(),
    timings_ms: z.record(z.string(), z.number()).optional(),
  }),
  teeth: z.array(rawToothSchema),
});

const surfaceNames = new Set(['mesial', 'distal', 'occlusal']);
const surfaceMethods = new Set(['RF', 'XThirds_Fallback']);

export interface ApiInferenceData {
  meta: {
    job_id: number;
    processed_at: string;
    models: { detector: string; classifier: string };
    timings_ms: Record<string, number>;
  };
  image: { width: number; height: number };
  teeth: Array<{
    id: number;
    fdi: number;
    confidence: number;
    bbox: [number, number, number, number];
    mask: { encoding: 'polygon'; data: [number, number][] };
    axes: {
      major: [number, number];
      minor: [number, number];
      rotation_deg: number;
      clamped: boolean;
    };
    has_caries: boolean;
    surfaces: Array<{
      name: 'mesial' | 'distal' | 'occlusal';
      label: 'caries';
      probability: number | null;
      method: 'RF' | 'XThirds_Fallback';
    }>;
  }>;
}

function fallbackMask(
  bbox: [number, number, number, number]
): [number, number][] {
  const [x, y, width, height] = bbox;
  return [
    [x, y],
    [x + width, y],
    [x + width, y + height],
    [x, y + height],
  ];
}

function normalizeSurfaceName(name: string): 'mesial' | 'distal' | 'occlusal' {
  const normalized = name.toLowerCase();
  if (!surfaceNames.has(normalized)) {
    throw new Error(`Unsupported surface name: ${name}`);
  }
  return normalized as 'mesial' | 'distal' | 'occlusal';
}

function normalizeProcessedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) throw new Error('Invalid completed_at timestamp');
  return date.toISOString();
}

export function adaptInferenceResult(raw: unknown): ApiInferenceData {
  const parsed = rawResultSchema.parse(raw);
  const teeth = parsed.teeth.map((tooth, index) => {
    const surfaces = new Map<
      'mesial' | 'distal' | 'occlusal',
      ApiInferenceData['teeth'][number]['surfaces'][number]
    >();

    for (const surface of tooth.surfaces) {
      const name = normalizeSurfaceName(surface.name);
      const method = surfaceMethods.has(surface.method ?? '')
        ? (surface.method as 'RF' | 'XThirds_Fallback')
        : (() => {
            throw new Error(`Unsupported surface method: ${surface.method ?? 'missing'}`);
          })();
      const finding = {
        name,
        label: 'caries' as const,
        probability: method === 'RF' ? (surface.probability ?? null) : null,
        method,
      };
      const previous = surfaces.get(name);
      if (!previous || (finding.probability ?? -1) > (previous.probability ?? -1)) {
        surfaces.set(name, finding);
      }
    }

    const mask =
      tooth.mask.data.length >= 3 ? tooth.mask.data : fallbackMask(tooth.bbox);
    return {
      id: index,
      fdi: tooth.fdi,
      confidence: tooth.confidence,
      bbox: tooth.bbox,
      mask: { encoding: 'polygon' as const, data: mask },
      axes: {
        major: tooth.axes?.major ?? [0, 0],
        minor: tooth.axes?.minor ?? [0, 0],
        rotation_deg: tooth.axes?.rotation_deg ?? 0,
        clamped: tooth.axes?.clamped ?? false,
      },
      has_caries: surfaces.size > 0,
      surfaces: [...surfaces.values()],
    };
  });

  return {
    meta: {
      job_id: parsed.meta.job_id,
      processed_at: normalizeProcessedAt(parsed.meta.completed_at),
      models: parsed.meta.models ?? { detector: 'unknown', classifier: 'unknown' },
      timings_ms: parsed.meta.timings_ms ?? {},
    },
    image: parsed.meta.image_size,
    teeth,
  };
}
