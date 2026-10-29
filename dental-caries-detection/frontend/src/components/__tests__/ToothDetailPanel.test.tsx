import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import type { ToothViewModel } from '../../features/analysis/analysisTypes';
import { ToothDetailPanel } from '../ToothDetailPanel';

afterEach(cleanup);

function toothWithProbability(probability: number | null): ToothViewModel {
  return {
    id: 0,
    fdi: 46,
    displayLabel: 'FDI 46',
    confidence: 0.73,
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
    cariesCount: 1,
    cariesSummary: '1 caries surface',
    hasCaries: true,
    colorKey: '#2563eb',
  };
}

describe('ToothDetailPanel', () => {
  it('renders an em dash when the surface probability is null', () => {
    render(<ToothDetailPanel tooth={toothWithProbability(null)} onClose={() => {}} />);

    expect(screen.getByRole('cell', { name: '—' })).not.toBeNull();
    expect(screen.queryByText('N/A')).toBeNull();
  });

  it('renders numeric surface probabilities as percentages', () => {
    render(<ToothDetailPanel tooth={toothWithProbability(0.73)} onClose={() => {}} />);

    expect(screen.getByRole('cell', { name: '73%' })).not.toBeNull();
  });
});
