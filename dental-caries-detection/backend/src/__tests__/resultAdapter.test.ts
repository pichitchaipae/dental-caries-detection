import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { adaptInferenceResult } from '../services/resultAdapter.js';

const raw = {
  meta: {
    job_id: 7,
    completed_at: '2026-10-28T10:00:00Z',
    image_size: { width: 1000, height: 500 },
    models: { detector: 'Tooth_seg_pano_20250319.pt', classifier: 'not_configured' },
    timings_ms: { stage1: 12.5 },
  },
  teeth: [
    {
      id: 0,
      fdi: 46,
      confidence: 0.91,
      bbox: [10, 20, 40, 80],
      mask: {
        encoding: 'polygon',
        data: [
          [10, 20],
          [50, 20],
          [50, 100],
          [10, 100],
        ],
      },
      surfaces: [
        {
          name: 'Distal',
          label: 'caries',
          probability: 0.77,
          method: 'Geometry_Heuristic',
          x_rel: 0.12,
        },
      ],
      caries: [
        {
          bbox: [11, 30, 8, 12],
          mask: {
            encoding: 'polygon',
            data: [
              [11, 30],
              [19, 30],
              [15, 42],
            ],
          },
          confidence: 0.77,
          centroid: [15, 34],
        },
      ],
    },
  ],
};

describe('adaptInferenceResult', () => {
  it('passes Geometry_Heuristic detector confidence through as probability', () => {
    const data = adaptInferenceResult(raw);
    expect(data.teeth[0].surfaces).toEqual([
      { name: 'distal', label: 'caries', probability: 0.77, method: 'Geometry_Heuristic' },
    ]);
    expect(data.teeth[0].has_caries).toBe(true);
  });

  it('maps legacy lesions to v2 without leaking confidence or bbox', () => {
    const data = adaptInferenceResult(raw);
    expect(data.teeth[0]).not.toHaveProperty('confidence');
    expect(data.teeth[0].caries[0]).toEqual({
      id: 0,
      surface: null,
      probability: 0.77,
      mask: raw.teeth[0].caries[0].mask,
      centroid: [15, 34],
    });
    expect(data.teeth[0].caries).toHaveLength(1);
    expect(data.teeth[0].caries[0].centroid).toEqual([15, 34]);
    expect(data.teeth[0].caries[0].mask.data).toHaveLength(3);
  });

  it('defaults missing axes and caries for older results', () => {
    const legacy = {
      ...raw,
      teeth: [{ ...raw.teeth[0], axes: undefined, caries: undefined }],
    };
    const data = adaptInferenceResult(legacy);
    expect(data.teeth[0].axes).toEqual({
      major: [0, 0],
      minor: [0, 0],
      rotation_deg: 0,
      clamped: false,
    });
    expect(data.teeth[0].caries).toEqual([]);
  });

  it('rejects an unknown surface method', () => {
    const bad = {
      ...raw,
      teeth: [
        {
          ...raw.teeth[0],
          surfaces: [{ ...raw.teeth[0].surfaces[0], method: 'magic' }],
        },
      ],
    };
    expect(() => adaptInferenceResult(bad)).toThrow('Unsupported surface method');
  });
});

describe('v2 lesion contract', () => {
  const lesion = raw.teeth[0].caries[0];
  function withLesions(caries: unknown[], surfaces = raw.teeth[0].surfaces) {
    return { ...raw, teeth: [{ ...raw.teeth[0], caries, surfaces }] };
  }

  it('returns empty caries and no findings for a sound tooth', () => {
    const data = adaptInferenceResult(withLesions([], []));
    expect(data.teeth[0].caries).toEqual([]);
    expect(data.teeth[0].has_caries).toBe(false);
  });

  it('preserves ML ids and surfaces for multiple lesions on one surface', () => {
    const data = adaptInferenceResult(
      withLesions([
        { ...lesion, id: 3, surface: 'distal', confidence: 0.5 },
        { ...lesion, id: 7, surface: 'distal' },
      ])
    );
    expect(
      data.teeth[0].caries.map(({ id, surface, probability }) => ({ id, surface, probability }))
    ).toEqual([
      { id: 3, surface: 'distal', probability: 0.5 },
      { id: 7, surface: 'distal', probability: 0.77 },
    ]);
    expect(data.teeth[0].surfaces).toHaveLength(1);
    expect(data.teeth[0].surfaces[0].probability).toBe(0.77);
  });

  it('allocates fallback ids without colliding with ML ids', () => {
    const data = adaptInferenceResult(withLesions([lesion, { ...lesion, id: 0 }, lesion]));
    expect(data.teeth[0].caries.map(({ id }) => id)).toEqual([1, 0, 2]);
  });

  it('rejects duplicate ML ids', () => {
    expect(() =>
      adaptInferenceResult(
        withLesions([
          { ...lesion, id: 0 },
          { ...lesion, id: 0 },
        ])
      )
    ).toThrow('Duplicate lesion id');
  });

  it('logs a surface mismatch but retains the result', () => {
    const logger = { warn: vi.fn() };
    const data = adaptInferenceResult(
      withLesions([{ ...lesion, id: 2, surface: 'mesial' }]),
      logger
    );
    expect(data.teeth[0].caries[0].surface).toBe('mesial');
    expect(logger.warn).toHaveBeenCalledWith(
      { job_id: 7, tooth_id: 0, lesion_id: 2, surface: 'mesial' },
      'Lesion surface is absent from tooth surfaces'
    );
  });

  it('accepts null or missing surfaces while ML-10 is pending', () => {
    const logger = { warn: vi.fn() };
    const data = adaptInferenceResult(withLesions([lesion, { ...lesion, surface: null }]), logger);
    expect(data.teeth[0].caries.map(({ surface }) => surface)).toEqual([null, null]);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it.each([-0.1, 1.1])('rejects out-of-range lesion confidence %s', (confidence) => {
    expect(() => adaptInferenceResult(withLesions([{ ...lesion, confidence }]))).toThrow();
  });

  it('matches the public fixture to the raw fixture', () => {
    const load = (name: string) =>
      JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8')) as unknown;
    const sample = load('result.sample.json') as { data: unknown };
    expect(adaptInferenceResult(load('ml-result.raw.sample.json'))).toEqual(sample.data);
  });
});

describe('recorded ML fixture', () => {
  it('retains both real lesion polygons in original panorama coordinates', () => {
    const rawFixture: unknown = JSON.parse(
      readFileSync(new URL('../fixtures/ml-result.raw.sample.json', import.meta.url), 'utf8')
    );
    const data = adaptInferenceResult(rawFixture);
    expect(data.image).toEqual({ width: 3037, height: 1536 });
    expect(data.teeth).toHaveLength(29);
    const affected = data.teeth.filter((tooth) => tooth.caries.length > 0);
    expect(affected.map((tooth) => tooth.fdi)).toEqual([45, 26]);
    expect(affected.map((tooth) => tooth.caries[0].mask.data.length)).toEqual([20, 59]);
    expect(affected.map((tooth) => tooth.caries[0].probability)).toEqual([0.3152, 0.5933]);
    for (const tooth of affected) {
      const lesion = tooth.caries[0];
      expect(lesion.surface).toBe('distal');
      expect(tooth.surfaces.find((surface) => surface.name === lesion.surface)?.probability).toBe(
        lesion.probability
      );
      expect(lesion).not.toHaveProperty('bbox');
      expect(lesion).not.toHaveProperty('confidence');
      for (const [x, y] of lesion.mask.data) {
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThanOrEqual(data.image.width);
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThanOrEqual(data.image.height);
      }
    }
  });
});
