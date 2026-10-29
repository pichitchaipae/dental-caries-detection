import { describe, expect, it } from 'vitest';
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

  it('passes the additive caries lesion array through', () => {
    const data = adaptInferenceResult(raw);
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
