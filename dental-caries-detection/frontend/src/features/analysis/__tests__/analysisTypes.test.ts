import { describe, expect, it } from 'vitest';
import { toViewModel } from '../analysisTypes';
import type { InferenceData } from '../../../domain/inference';
import fixture from '../../../fixtures/result.sample.json';

// The JSON import's inferred types (e.g. `bbox: number[]`) are wider than the
// tuple-typed `InferenceData`; this fixture is already validated against the
// real schema by domain/__tests__/inference.test.ts, so the cast is safe here.
const data = fixture.data as unknown as InferenceData;

describe('toViewModel', () => {
  it('computes display labels, caries summaries, and a summary block from the fixture', () => {
    const vm = toViewModel(data);

    expect(vm.image).toEqual(data.image);
    expect(vm.teeth).toHaveLength(3);

    const tooth36 = vm.teeth.find((t) => t.fdi === 36);
    expect(tooth36?.displayLabel).toBe('FDI 36');
    expect(tooth36?.cariesCount).toBe(1);
    expect(tooth36?.cariesSummary).toBe('1 / 5 surfaces');
    expect(tooth36?.hasCaries).toBe(true);

    const tooth16 = vm.teeth.find((t) => t.fdi === 16);
    expect(tooth16?.hasCaries).toBe(false);
    expect(tooth16?.cariesSummary).toBe('0 / 5 surfaces');

    expect(vm.summary).toEqual({
      totalTeeth: 3,
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
    expect(vm.teeth[0].cariesSummary).toBe('0 / 0 surfaces');
    expect(vm.teeth[0].hasCaries).toBe(false);
  });

  it('handles a tooth where every surface has caries', () => {
    const allCaries: InferenceData = {
      ...data,
      teeth: [
        {
          ...data.teeth[0],
          surfaces: data.teeth[0].surfaces.map((s) => ({ ...s, label: 'caries' as const })),
        },
      ],
    };
    const vm = toViewModel(allCaries);

    expect(vm.teeth[0].cariesCount).toBe(5);
    expect(vm.teeth[0].hasCaries).toBe(true);
  });
});
