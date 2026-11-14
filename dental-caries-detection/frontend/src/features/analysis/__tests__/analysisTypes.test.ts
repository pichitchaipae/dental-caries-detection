import { describe, expect, it } from 'vitest';
import { toViewModel } from '../analysisTypes';
import type { InferenceData } from '../../../domain/inference';
import { loadV2Data } from '../../../test-utils/v2Fixture';

const data: InferenceData = loadV2Data();

describe('toViewModel', () => {
  it('computes display labels, caries summaries, and a summary block from the v2 fixture', () => {
    const vm = toViewModel(data);

    expect(vm.image).toEqual(data.image);
    expect(vm.teeth).toHaveLength(29);

    const tooth45 = vm.teeth.find((t) => t.fdi === 45);
    expect(tooth45?.displayLabel).toBe('FDI 45');
    expect(tooth45?.cariesCount).toBe(1);
    expect(tooth45?.cariesSummary).toBe('1 caries surface');
    expect(tooth45?.hasCaries).toBe(true);

    const tooth11 = vm.teeth.find((t) => t.fdi === 11);
    expect(tooth11?.hasCaries).toBe(false);
    expect(tooth11?.cariesSummary).toBe('0 caries surfaces');

    expect(vm.summary).toEqual({
      totalTeeth: 29,
      teethWithCaries: 2, // the fixture has predicted caries on FDI 45 and FDI 26
      totalCariesSurfaces: 2,
    });
  });

  it('forwards caries[] (id, surface, probability, mask, centroid) to the view model', () => {
    const vm = toViewModel(data);
    const tooth26 = vm.teeth.find((t) => t.fdi === 26);
    const source = data.teeth.find((t) => t.fdi === 26);

    expect(tooth26?.caries).toHaveLength(1);
    expect(tooth26?.caries).toEqual(source?.caries);
    expect(tooth26?.caries[0].surface).toBe('distal');
    expect(tooth26?.caries[0].probability).toBe(0.5933);
    expect(tooth26?.caries[0].mask.data).toHaveLength(59);
    expect(tooth26?.caries[0].centroid).toEqual([2067.43, 732.34]);
  });

  it('gives a sound tooth an empty caries array', () => {
    const vm = toViewModel(data);
    const sound = vm.teeth.filter((t) => !t.hasCaries);

    expect(sound.length).toBeGreaterThan(0);
    expect(sound.every((t) => Array.isArray(t.caries) && t.caries.length === 0)).toBe(true);
  });

  it('no longer exposes the tooth detection confidence', () => {
    const vm = toViewModel(data);

    expect(vm.teeth.every((tooth) => !('confidence' in tooth))).toBe(true);
    // bbox stays: it anchors the label and the hit-test fallback.
    expect(vm.teeth[0].bbox).toEqual(data.teeth[0].bbox);
  });

  it('carries a lesion whose surface is null through unchanged', () => {
    const withNullSurface: InferenceData = {
      ...data,
      teeth: data.teeth.map((tooth) =>
        tooth.caries.length > 0
          ? { ...tooth, caries: tooth.caries.map((lesion) => ({ ...lesion, surface: null })) }
          : tooth
      ),
    };
    const vm = toViewModel(withNullSurface);

    expect(vm.teeth.find((t) => t.fdi === 45)?.caries[0].surface).toBeNull();
    expect(vm.summary.totalCariesSurfaces).toBe(2); // counts surfaces[], unaffected by lesion surface
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
      teeth: [{ ...data.teeth[0], surfaces: [], caries: [] }],
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
            method: 'Geometry_Heuristic' as const,
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
