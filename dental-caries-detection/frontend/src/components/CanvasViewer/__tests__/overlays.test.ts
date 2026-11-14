import { describe, expect, it, vi } from 'vitest';
import * as overlays from '../overlays';
import { toViewModel } from '../../../features/analysis/analysisTypes';
import { loadV2Data } from '../../../test-utils/v2Fixture';

const { teeth, image } = (() => {
  const vm = toViewModel(loadV2Data());
  return { teeth: vm.teeth, image: vm.image };
})();
const toothWithCaries = teeth.find((t) => t.fdi === 26)!; // one lesion, 59-point polygon
const soundTooth = teeth.find((t) => t.fdi === 11)!;

// Minimal recording stand-in for CanvasRenderingContext2D (jsdom has no canvas).
function makeCtx() {
  const fills: unknown[] = [];
  const ctx = {
    fillStyle: '' as unknown,
    strokeStyle: '' as unknown,
    lineWidth: 0,
    lineJoin: '',
    shadowColor: '',
    shadowBlur: 0,
    save: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    fill: vi.fn(() => fills.push(ctx.fillStyle)),
    stroke: vi.fn(),
    strokeRect: vi.fn(),
  };
  return { ctx, fills, asCtx: ctx as unknown as CanvasRenderingContext2D };
}

describe('overlays', () => {
  it('no longer exports a bounding-box primitive', () => {
    expect(Object.keys(overlays)).not.toContain('drawBoundingBox');
  });

  describe('drawCariesMasks', () => {
    it('fills and outlines each lesion polygon in the caries mask color', () => {
      const { ctx, fills, asCtx } = makeCtx();
      overlays.drawCariesMasks(asCtx, toothWithCaries);

      const lesion = toothWithCaries.caries[0];
      expect(ctx.beginPath).toHaveBeenCalledTimes(1);
      expect(ctx.moveTo).toHaveBeenCalledWith(...lesion.mask.data[0]);
      expect(ctx.lineTo).toHaveBeenCalledTimes(lesion.mask.data.length - 1);
      expect(ctx.lineTo).toHaveBeenLastCalledWith(...lesion.mask.data.at(-1)!);
      expect(ctx.closePath).toHaveBeenCalledTimes(1);
      expect(fills).toEqual([overlays.CARIES_MASK_FILL]);
      expect(ctx.stroke).toHaveBeenCalledTimes(1);
      expect(ctx.strokeStyle).toBe(overlays.CARIES_MASK_STROKE);
    });

    it('draws one path per lesion when a tooth has several', () => {
      const second = { ...toothWithCaries.caries[0], id: 1 };
      const tooth = { ...toothWithCaries, caries: [toothWithCaries.caries[0], second] };
      const { ctx, fills, asCtx } = makeCtx();
      overlays.drawCariesMasks(asCtx, tooth);

      expect(ctx.beginPath).toHaveBeenCalledTimes(2);
      expect(fills).toHaveLength(2);
    });

    it.each([0, 1, 2])('skips a lesion polygon with only %i point(s) without throwing', (n) => {
      const short = {
        ...toothWithCaries.caries[0],
        id: 1,
        mask: {
          encoding: 'polygon' as const,
          data: toothWithCaries.caries[0].mask.data.slice(0, n),
        },
      };
      const tooth = { ...toothWithCaries, caries: [short, toothWithCaries.caries[0]] };
      const { ctx, fills, asCtx } = makeCtx();

      expect(() => overlays.drawCariesMasks(asCtx, tooth)).not.toThrow();
      // Only the valid lesion is traced and filled.
      expect(ctx.beginPath).toHaveBeenCalledTimes(1);
      expect(fills).toEqual([overlays.CARIES_MASK_FILL]);
    });

    it('draws nothing for a tooth without caries[]', () => {
      const { ctx, asCtx } = makeCtx();
      overlays.drawCariesMasks(asCtx, soundTooth);

      expect(ctx.beginPath).not.toHaveBeenCalled();
      expect(ctx.fill).not.toHaveBeenCalled();
    });

    it('restores the context state it saved', () => {
      const { ctx, asCtx } = makeCtx();
      overlays.drawCariesMasks(asCtx, toothWithCaries);

      expect(ctx.save).toHaveBeenCalledTimes(ctx.restore.mock.calls.length);
    });
  });

  describe('drawMask', () => {
    it('keeps the tooth fill distinct from the caries mask fill', () => {
      const { fills, asCtx } = makeCtx();
      overlays.drawMask(asCtx, toothWithCaries, image.width, image.height);

      expect(fills).toHaveLength(1);
      expect(fills[0]).not.toBe(overlays.CARIES_MASK_FILL);
    });
  });

  describe('drawSelectionHalo', () => {
    it('outlines the tooth mask polygon and never strokes a bounding rectangle', () => {
      const { ctx, asCtx } = makeCtx();
      overlays.drawSelectionHalo(asCtx, toothWithCaries, image.width, image.height);

      expect(ctx.strokeRect).not.toHaveBeenCalled();
      expect(ctx.moveTo).toHaveBeenCalledWith(
        ...(toothWithCaries.mask.data as [number, number][])[0]
      );
      expect(ctx.stroke).toHaveBeenCalledTimes(1);
    });
  });
});
