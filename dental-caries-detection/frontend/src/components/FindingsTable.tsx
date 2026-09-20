import type { ToothViewModel } from '../features/analysis/analysisTypes';

interface FindingsTableProps {
  teeth: ToothViewModel[];
  selectedToothId: number | null;
  onSelectTooth: (id: number) => void;
}

export function FindingsTable({ teeth, selectedToothId, onSelectTooth }: FindingsTableProps) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <caption className="px-6 pb-2 pt-4 text-left font-semibold text-slate-900">
          All detected teeth
        </caption>
        <thead>
          <tr className="text-left text-slate-500">
            <th className="border-b border-slate-200 px-6 py-2 font-medium">FDI</th>
            <th className="border-b border-slate-200 px-6 py-2 font-medium">Confidence</th>
            <th className="border-b border-slate-200 px-6 py-2 font-medium">Caries surfaces</th>
          </tr>
        </thead>
        <tbody>
          {teeth.length === 0 ? (
            <tr>
              <td colSpan={3} className="px-6 py-4 text-center text-slate-500">
                No teeth detected in this image.
              </td>
            </tr>
          ) : (
            teeth.map((tooth) => {
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
