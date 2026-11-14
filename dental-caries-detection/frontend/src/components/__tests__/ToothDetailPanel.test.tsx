import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen, within } from '@testing-library/react';
import type { ToothViewModel } from '../../features/analysis/analysisTypes';
import type { CariesLesion } from '../../domain/inference';
import { ToothDetailPanel } from '../ToothDetailPanel';

afterEach(cleanup);

const squareMask: CariesLesion['mask'] = {
  encoding: 'polygon',
  data: [
    [12, 30],
    [20, 30],
    [20, 40],
    [12, 40],
  ],
};

function lesion(overrides: Partial<CariesLesion> = {}): CariesLesion {
  return {
    id: 0,
    surface: 'occlusal',
    probability: 0.61,
    mask: squareMask,
    centroid: [16, 35],
    ...overrides,
  };
}

function toothWith(probability: number | null, caries: CariesLesion[] = []): ToothViewModel {
  return {
    id: 0,
    fdi: 46,
    displayLabel: 'FDI 46',
    bbox: [10, 20, 40, 80],
    mask: {
      encoding: 'polygon',
      data: [
        [10, 20],
        [50, 20],
        [50, 100],
      ],
    },
    axes: { major: [0, 0], minor: [0, 0], rotation_deg: 0, clamped: false },
    surfaces: [{ name: 'occlusal', label: 'caries', probability, method: 'Geometry_Heuristic' }],
    caries,
    cariesCount: 1,
    cariesSummary: '1 caries surface',
    hasCaries: true,
    colorKey: '#2563eb',
  };
}

describe('ToothDetailPanel', () => {
  it('renders an em dash when the surface probability is null', () => {
    render(<ToothDetailPanel tooth={toothWith(null)} onClose={() => {}} />);

    expect(screen.getByRole('cell', { name: '—' })).not.toBeNull();
    expect(screen.queryByText('N/A')).toBeNull();
  });

  it('renders numeric surface probabilities as percentages', () => {
    render(<ToothDetailPanel tooth={toothWith(0.73)} onClose={() => {}} />);

    expect(screen.getByRole('cell', { name: '73%' })).not.toBeNull();
  });

  it('no longer shows the tooth detection confidence', () => {
    render(<ToothDetailPanel tooth={toothWith(0.73)} onClose={() => {}} />);

    expect(screen.queryByText(/detection confidence/i)).toBeNull();
    expect(screen.getByText('1 caries surface')).not.toBeNull();
  });

  it('omits the lesion table when the tooth has no caries[]', () => {
    render(<ToothDetailPanel tooth={toothWith(0.73, [])} onClose={() => {}} />);

    expect(screen.queryByRole('table', { name: 'Caries lesions' })).toBeNull();
  });

  it('lists each lesion with its surface and probability', () => {
    const tooth = toothWith(0.73, [
      lesion({ id: 0, surface: 'occlusal', probability: 0.73 }),
      lesion({ id: 1, surface: 'distal', probability: 0.42 }),
    ]);
    render(<ToothDetailPanel tooth={tooth} onClose={() => {}} />);

    const table = screen.getByRole('table', { name: 'Caries lesions' });
    const rows = within(table).getAllByRole('row').slice(1);
    expect(rows.map((row) => row.textContent)).toEqual(['#1occlusal73%', '#2distal42%']);
  });

  it('shows "Unspecified" instead of crashing when a lesion surface is null', () => {
    const tooth = toothWith(0.73, [lesion({ id: 0, surface: null, probability: 0.55 })]);
    render(<ToothDetailPanel tooth={tooth} onClose={() => {}} />);

    const table = screen.getByRole('table', { name: 'Caries lesions' });
    expect(within(table).getByRole('cell', { name: 'Unspecified' })).not.toBeNull();
    expect(within(table).getByRole('cell', { name: '55%' })).not.toBeNull();
  });
});
