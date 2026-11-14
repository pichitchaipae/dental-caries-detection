import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// vite.config.ts runs Vitest without `globals: true`, so @testing-library/react's
// auto-cleanup (which hooks the global `afterEach`) never attaches; unmount
// explicitly so each test starts from an empty document.
afterEach(cleanup);
import { FindingsTable } from '../FindingsTable';
import { toViewModel } from '../../features/analysis/analysisTypes';
import { loadV2Data } from '../../test-utils/v2Fixture';

const teeth = toViewModel(loadV2Data()).teeth; // 29 teeth; caries on FDI 26 and 45 only
const allFdiAsc = teeth.map((tooth) => tooth.fdi).sort((a, b) => a - b);

function fdiColumn() {
  return screen
    .getAllByRole('row')
    .slice(1)
    .map((row) => row.querySelector('td')?.textContent?.trim());
}

describe('FindingsTable', () => {
  it('defaults to caries count desc, then FDI asc', () => {
    render(<FindingsTable teeth={teeth} selectedToothId={null} onSelectTooth={vi.fn()} />);
    // FDI 26 and 45 each have 1 caries surface (tie -> FDI asc: 26 before 45); the rest have 0, sort after by FDI asc.
    const rest = allFdiAsc.filter((fdi) => fdi !== 26 && fdi !== 45);
    expect(fdiColumn()).toEqual([26, 45, ...rest].map(String));
  });

  it('sorts by FDI ascending, then descending on repeat click', async () => {
    const user = userEvent.setup();
    render(<FindingsTable teeth={teeth} selectedToothId={null} onSelectTooth={vi.fn()} />);

    await user.click(screen.getByRole('columnheader', { name: /FDI/ }));
    expect(fdiColumn()).toEqual(allFdiAsc.map(String));

    await user.click(screen.getByRole('columnheader', { name: /FDI/ }));
    expect(fdiColumn()).toEqual([...allFdiAsc].reverse().map(String));
  });

  it('no longer shows a tooth detection confidence column', () => {
    render(<FindingsTable teeth={teeth} selectedToothId={null} onSelectTooth={vi.fn()} />);

    expect(screen.queryByRole('columnheader', { name: /confidence/i })).toBeNull();
    expect(screen.getAllByRole('columnheader').map((th) => th.textContent?.trim())).toEqual([
      'FDI',
      'Caries surfaces ↓',
    ]);
    // 26 and 45 show their surface chip; every other row shows the em dash.
    const firstRowCells = screen.getAllByRole('row')[1].querySelectorAll('td');
    expect(firstRowCells).toHaveLength(2);
    expect(firstRowCells[1].textContent).toBe('distal');
  });

  it('shows an empty state when there are no teeth', () => {
    render(<FindingsTable teeth={[]} selectedToothId={null} onSelectTooth={vi.fn()} />);
    // getByText throws (failing the test) if the empty-state message isn't rendered.
    expect(screen.getByText('No teeth detected in this image.')).not.toBeNull();
  });
});
