import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// vite.config.ts runs Vitest without `globals: true`, so @testing-library/react's
// auto-cleanup (which hooks the global `afterEach`) never attaches; unmount
// explicitly so each test starts from an empty document.
afterEach(cleanup);
import { FindingsTable } from '../FindingsTable';
import { toViewModel } from '../../features/analysis/analysisTypes';
import type { InferenceData } from '../../domain/inference';
import fixture from '../../fixtures/result.sample.json';

const data = fixture.data as unknown as InferenceData;
const teeth = toViewModel(data).teeth; // FDI 16 (0 caries), 36 (1 caries), 46 (1 caries)

function fdiColumn() {
  return screen
    .getAllByRole('row')
    .slice(1)
    .map((row) => row.querySelector('td')?.textContent?.trim());
}

describe('FindingsTable', () => {
  it('defaults to caries count desc, then FDI asc', () => {
    render(<FindingsTable teeth={teeth} selectedToothId={null} onSelectTooth={vi.fn()} />);
    // FDI 36 and 46 both have 1 caries surface (tie -> FDI asc: 36 before 46); FDI 16 has 0, sorts last.
    expect(fdiColumn()).toEqual(['36', '46', '16']);
  });

  it('sorts by FDI ascending, then descending on repeat click', async () => {
    const user = userEvent.setup();
    render(<FindingsTable teeth={teeth} selectedToothId={null} onSelectTooth={vi.fn()} />);

    await user.click(screen.getByRole('columnheader', { name: /FDI/ }));
    expect(fdiColumn()).toEqual(['16', '36', '46']);

    await user.click(screen.getByRole('columnheader', { name: /FDI/ }));
    expect(fdiColumn()).toEqual(['46', '36', '16']);
  });

  it('shows an empty state when there are no teeth', () => {
    render(<FindingsTable teeth={[]} selectedToothId={null} onSelectTooth={vi.fn()} />);
    // getByText throws (failing the test) if the empty-state message isn't rendered.
    expect(screen.getByText('No teeth detected in this image.')).not.toBeNull();
  });
});
