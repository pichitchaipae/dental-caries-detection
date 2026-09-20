import type { Tooth } from '../domain/inference';

interface FindingsTableProps {
  teeth: Tooth[];
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
          {teeth.map((tooth) => {
            const cariesSurfaces = tooth.surfaces.filter((surface) => surface.label === 'caries');
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
                <td className="border-b border-slate-100 px-6 py-2.5">{tooth.fdi}</td>
                <td className="border-b border-slate-100 px-6 py-2.5">
                  {(tooth.confidence * 100).toFixed(0)}%
                </td>
                <td className="border-b border-slate-100 px-6 py-2.5">
                  {cariesSurfaces.length === 0
                    ? '—'
                    : cariesSurfaces.map((surface) => surface.name).join(', ')}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
