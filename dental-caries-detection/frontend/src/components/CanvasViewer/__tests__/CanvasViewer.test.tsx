import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CanvasViewer } from '../CanvasViewer';
import { CARIES_MASK_FILL } from '../overlays';
import { toViewModel } from '../../../features/analysis/analysisTypes';
import { loadV2Data } from '../../../test-utils/v2Fixture';

afterEach(cleanup);

const vm = toViewModel(loadV2Data());
const fills: unknown[] = [];
let frames = 0; // one clearRect per drawn frame

// jsdom ships no canvas, ResizeObserver or HTMLImageElement.decode; stub just
// enough for CanvasViewer to mount and run its (rAF-coalesced) draw loop
// against a recording context.
beforeEach(() => {
  fills.length = 0;
  frames = 0;
  const ctx: Record<string, unknown> = {
    fillStyle: '',
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    clearRect: vi.fn(() => frames++),
    drawImage: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    strokeRect: vi.fn(),
    strokeText: vi.fn(),
    fillText: vi.fn(),
    fill: vi.fn(() => fills.push(ctx.fillStyle)),
  };
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
      unobserve() {}
    }
  );
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
    ctx as unknown as CanvasRenderingContext2D
  );
  Object.defineProperty(HTMLImageElement.prototype, 'decode', {
    configurable: true,
    value: () => Promise.resolve(),
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  delete (HTMLImageElement.prototype as { decode?: unknown }).decode;
});

function renderViewer(teeth = vm.teeth) {
  return render(
    <CanvasViewer
      imageBase64="data:image/png;base64,AAAA"
      imageSize={vm.image}
      teeth={teeth}
      selectedToothId={null}
      onSelectTooth={vi.fn()}
    />
  );
}

const lesionFillCount = () => fills.filter((f) => f === CARIES_MASK_FILL).length;

describe('CanvasViewer layer controls', () => {
  it('offers Masks, Caries Mask, Axes and Labels toggles, and no Boxes toggle', () => {
    renderViewer();

    expect(screen.getAllByRole('checkbox').map((box) => box.parentElement?.textContent)).toEqual([
      'Masks',
      'Caries Mask',
      'Axes',
      'Labels',
    ]);
    expect(screen.queryByLabelText('Boxes')).toBeNull();
  });

  it('has the Caries Mask layer on by default', () => {
    renderViewer();

    expect((screen.getByLabelText('Caries Mask') as HTMLInputElement).checked).toBe(true);
  });

  it('draws one lesion fill per caries mask, and stops when the layer is switched off', async () => {
    const user = userEvent.setup();
    renderViewer();

    // The fixture has two lesions (FDI 45 and 26).
    await waitFor(() => expect(lesionFillCount()).toBeGreaterThanOrEqual(2));
    expect(lesionFillCount() % 2).toBe(0);

    await user.click(screen.getByLabelText('Caries Mask'));
    expect((screen.getByLabelText('Caries Mask') as HTMLInputElement).checked).toBe(false);

    // Let the redraw triggered by the toggle run, then check that a fresh frame has no lesion fills.
    fills.length = 0;
    await waitFor(() => expect(fills.length).toBeGreaterThan(0)); // tooth masks still drawn
    expect(lesionFillCount()).toBe(0);

    await user.click(screen.getByLabelText('Caries Mask'));
    fills.length = 0;
    await waitFor(() => expect(lesionFillCount()).toBe(2));
  });

  it('keeps drawing lesions when only the tooth Masks layer is switched off', async () => {
    const user = userEvent.setup();
    renderViewer();
    await waitFor(() => expect(lesionFillCount()).toBeGreaterThanOrEqual(2));

    await user.click(screen.getByLabelText('Masks'));
    fills.length = 0;
    await waitFor(() => expect(lesionFillCount()).toBe(2));
    expect(fills.every((f) => f === CARIES_MASK_FILL)).toBe(true);
  });

  it('does not crash on a lesion with fewer than 3 mask points and draws only the valid lesion', async () => {
    const degenerate = vm.teeth.map((tooth) =>
      tooth.fdi === 45
        ? {
            ...tooth,
            caries: tooth.caries.map((lesion) => ({
              ...lesion,
              mask: { ...lesion.mask, data: lesion.mask.data.slice(0, 2) },
            })),
          }
        : tooth
    );
    renderViewer(degenerate);

    // FDI 45's lesion is skipped; each drawn frame fills exactly one lesion (FDI 26's).
    await waitFor(() => expect(frames).toBeGreaterThan(0));
    expect(lesionFillCount()).toBe(frames);
  });
});
