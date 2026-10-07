import { describe, expect, it } from 'vitest';
import { toViewModel } from '../analysisTypes';
import { parseProcessResponse, type InferenceData } from '../../../domain/inference';
import rawResult from '../../../fixtures/ml-result.raw.sample.json';

const apiData = {
  meta: {
    job_id: rawResult.meta.job_id,
    processed_at: new Date(rawResult.meta.completed_at).toISOString(),
    models: { detector: 'unknown', classifier: 'unknown' },
    timings_ms: {},
  },
  image: rawResult.meta.image_size,
  teeth: rawResult.teeth.map((tooth, id) => ({
    ...tooth,
    id,
    axes: { ...tooth.axes, clamped: tooth.axes.clamped ?? false },
    has_caries: tooth.surfaces.length > 0,
    surfaces: tooth.surfaces.map((surface) => ({
      ...surface,
      probability: surface.method === 'RF' ? surface.probability : null,
    })),
  })),
};
const parsed = parseProcessResponse({ status: 'done', image_base64: 'x', data: apiData });
if (parsed.status !== 'done') throw new Error('fixture must parse as a done response');
const data: InferenceData = parsed.data;

describe('toViewModel', () => {
  it('computes display labels, caries summaries, and a summary block from the fixture', () => {
    const vm = toViewModel(data);

    expect(vm.image).toEqual(data.image);
    expect(vm.teeth).toHaveLength(4);

    const tooth36 = vm.teeth.find((t) => t.fdi === 36);
    expect(tooth36?.displayLabel).toBe('FDI 36');
    expect(tooth36?.cariesCount).toBe(1);
    expect(tooth36?.cariesSummary).toBe('1 caries surface');
    expect(tooth36?.hasCaries).toBe(true);

    const tooth16 = vm.teeth.find((t) => t.fdi === 16);
    expect(tooth16?.hasCaries).toBe(false);
    expect(tooth16?.cariesSummary).toBe('0 caries surfaces');

    expect(vm.summary).toEqual({
      totalTeeth: 4,
      teethWithCaries: 2, // fixture has caries on FDI 36 and FDI 46
      totalCariesSurfaces: 2,
    });
  });

  it('assigns a stable, deterministic colorKey per tooth id', () => {
    const vm = toViewModel(data);
    const again = toViewModel(data);

    vm.teeth.forEach((tooth, i) => {
      expect(tooth.colorKey).toBe(again.teeth[i].colorKey);
      expect(tooth.colorKey).toMatch(/^#[0-9a-f]{6}$/);
    });
  });

  it('handles an empty teeth array', () => {
    const empty: InferenceData = { ...data, teeth: [] };
    const vm = toViewModel(empty);

    expect(vm.teeth).toEqual([]);
    expect(vm.summary).toEqual({ totalTeeth: 0, teethWithCaries: 0, totalCariesSurfaces: 0 });
  });

  it('handles a tooth with zero surfaces', () => {
    const noSurfaces: InferenceData = {
      ...data,
      teeth: [{ ...data.teeth[0], surfaces: [] }],
    };
    const vm = toViewModel(noSurfaces);

    expect(vm.teeth[0].cariesCount).toBe(0);
    expect(vm.teeth[0].cariesSummary).toBe('0 caries surfaces');
    expect(vm.teeth[0].hasCaries).toBe(false);
  });

  it('counts every caries surface on a tooth', () => {
    const names = ['mesial', 'distal', 'occlusal'] as const;
    const allCaries: InferenceData = {
      ...data,
      teeth: [
        {
          ...data.teeth[0],
          surfaces: names.map((name) => ({
            name,
            label: 'caries' as const,
            probability: 0.9,
            method: 'RF' as const,
          })),
        },
      ],
    };
    const vm = toViewModel(allCaries);

    expect(vm.teeth[0].cariesCount).toBe(3);
    expect(vm.teeth[0].cariesSummary).toBe('3 caries surfaces');
    expect(vm.teeth[0].hasCaries).toBe(true);
  });
});
