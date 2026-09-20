import { useMemo, useState } from 'react';
import type { ToothViewModel } from '../features/analysis/analysisTypes';

interface FindingsTableProps {
  teeth: ToothViewModel[];
  selectedToothId: number | null;
  onSelectTooth: (id: number) => void;
}

type SortColumn = 'fdi' | 'caries';
type SortDirection = 'asc' | 'desc';
interface SortState {
  column: SortColumn;
  direction: SortDirection;
}

// FE-6.6: default sort is caries count desc, then FDI asc; clicking a header
// re-sorts by that column, toggling direction on a repeat click.
const DEFAULT_SORT: SortState = { column: 'caries', direction: 'desc' };

function sortTeeth(teeth: ToothViewModel[], sort: SortState): ToothViewModel[] {
  const sorted = [...teeth];
  sorted.sort((a, b) => {
    if (sort.column === 'caries') {
      const diff =
        sort.direction === 'desc' ? b.cariesCount - a.cariesCount : a.cariesCount - b.cariesCount;
      return diff !== 0 ? diff : a.fdi - b.fdi; // tie-break: FDI asc, per the spec's default order
    }
    return sort.direction === 'asc' ? a.fdi - b.fdi : b.fdi - a.fdi;
  });
  return sorted;
}

function ariaSortFor(column: SortColumn, sort: SortState): 'ascending' | 'descending' | 'none' {
  if (sort.column !== column) return 'none';
  return sort.direction === 'asc' ? 'ascending' : 'descending';
}

export function FindingsTable({ teeth, selectedToothId, onSelectTooth }: FindingsTableProps) {
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);
  const sortedTeeth = useMemo(() => sortTeeth(teeth, sort), [teeth, sort]);

  const handleHeaderClick = (column: SortColumn) => {
    setSort((prev) =>
      prev.column === column
        ? { column, direction: prev.direction === 'asc' ? 'desc' : 'asc' }
        : { column, direction: column === 'caries' ? 'desc' : 'asc' }
    );
  };

  const sortIndicator = (column: SortColumn) => {
    if (sort.column !== column) return null;
    return <span aria-hidden="true">{sort.direction === 'asc' ? ' ↑' : ' ↓'}</span>;
  };

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <caption className="px-6 pb-2 pt-4 text-left font-semibold text-slate-900">
          All detected teeth
        </caption>
        <thead>
          <tr className="text-left text-slate-500">
            <th
              aria-sort={ariaSortFor('fdi', sort)}
              className="cursor-pointer select-none border-b border-slate-200 px-6 py-2 font-medium hover:text-slate-700"
              onClick={() => handleHeaderClick('fdi')}
            >
              FDI{sortIndicator('fdi')}
            </th>
            <th className="border-b border-slate-200 px-6 py-2 font-medium">Confidence</th>
            <th
              aria-sort={ariaSortFor('caries', sort)}
              className="cursor-pointer select-none border-b border-slate-200 px-6 py-2 font-medium hover:text-slate-700"
              onClick={() => handleHeaderClick('caries')}
            >
              Caries surfaces{sortIndicator('caries')}
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedTeeth.length === 0 ? (
            <tr>
              <td colSpan={3} className="px-6 py-4 text-center text-slate-500">
                No teeth detected in this image.
              </td>
            </tr>
          ) : (
            sortedTeeth.map((tooth) => {
              const cariesSurfaceNames = tooth.surfaces
                .filter((surface) => surface.label === 'caries')
                .map((surface) => surface.name)
                .join(', ');
              const isSelected = tooth.id === selectedToothId;

              return (
                <tr
                  key={tooth.id}
                  onClick={() => onSelectTooth(tooth.id)}
                  className={
                    'cursor-pointer transition-colors ' +
                    (isSelected ? 'bg-brand-50' : 'hover:bg-slate-50')
                  }
                >
                  <td className="border-b border-slate-100 px-6 py-2.5">
                    <span className="flex items-center gap-2">
                      <span
                        className="h-2 w-2 shrink-0 rounded-full"
                        style={{ backgroundColor: tooth.colorKey }}
                        aria-hidden="true"
                      />
                      {tooth.fdi}
                    </span>
                  </td>
                  <td className="border-b border-slate-100 px-6 py-2.5">
                    {(tooth.confidence * 100).toFixed(0)}%
                  </td>
                  <td className="border-b border-slate-100 px-6 py-2.5">
                    {tooth.hasCaries ? (
                      <span className="font-medium text-danger-600">{cariesSurfaceNames}</span>
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
