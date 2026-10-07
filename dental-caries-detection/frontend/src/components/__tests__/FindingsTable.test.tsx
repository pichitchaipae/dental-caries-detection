import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// vite.config.ts runs Vitest without `globals: true`, so @testing-library/react's
// auto-cleanup (which hooks the global `afterEach`) never attaches; unmount
// explicitly so each test starts from an empty document.
afterEach(cleanup);
import { FindingsTable } from '../FindingsTable';
import { toViewModel } from '../../features/analysis/analysisTypes';
import { parseProcessResponse } from '../../domain/inference';
import rawResult from '../../fixtures/ml-result.raw.sample.json';

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
const teeth = toViewModel(parsed.data).teeth; // FDI 11 & 16 (0 caries), 36 (1 caries), 46 (1 caries)

function fdiColumn() {
  return screen
    .getAllByRole('row')
    .slice(1)
    .map((row) => row.querySelector('td')?.textContent?.trim());
}

describe('FindingsTable', () => {
  it('defaults to caries count desc, then FDI asc', () => {
    render(<FindingsTable teeth={teeth} selectedToothId={null} onSelectTooth={vi.fn()} />);
    // FDI 36 and 46 both have 1 caries surface (tie -> FDI asc: 36 before 46); FDI 11 and 16 have 0, sort last.
    expect(fdiColumn()).toEqual(['36', '46', '11', '16']);
  });

  it('sorts by FDI ascending, then descending on repeat click', async () => {
    const user = userEvent.setup();
    render(<FindingsTable teeth={teeth} selectedToothId={null} onSelectTooth={vi.fn()} />);

    await user.click(screen.getByRole('columnheader', { name: /FDI/ }));
    expect(fdiColumn()).toEqual(['11', '16', '36', '46']);

    await user.click(screen.getByRole('columnheader', { name: /FDI/ }));
    expect(fdiColumn()).toEqual(['46', '36', '16', '11']);
  });

  it('shows an empty state when there are no teeth', () => {
    render(<FindingsTable teeth={[]} selectedToothId={null} onSelectTooth={vi.fn()} />);
    // getByText throws (failing the test) if the empty-state message isn't rendered.
    expect(screen.getByText('No teeth detected in this image.')).not.toBeNull();
  });
});
