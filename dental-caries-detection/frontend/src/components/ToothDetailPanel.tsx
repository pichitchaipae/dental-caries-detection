import type { Tooth } from '../domain/inference';

interface ToothDetailPanelProps {
  tooth: Tooth | null;
}

export function ToothDetailPanel({ tooth }: ToothDetailPanelProps) {
  if (!tooth) {
    return (
      <div className="flex h-full items-center rounded-xl border border-slate-200 bg-white p-6">
        <p className="text-sm text-slate-500">Select a tooth on the image to see its details.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6">
      <h2 className="text-base font-semibold text-slate-900">FDI {tooth.fdi}</h2>
      <p className="mb-3 text-sm text-slate-500">
        Detection confidence: {(tooth.confidence * 100).toFixed(0)}%
      </p>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="border-b border-slate-200 py-1.5 font-medium">Surface</th>
            <th className="border-b border-slate-200 py-1.5 font-medium">Finding</th>
            <th className="border-b border-slate-200 py-1.5 font-medium">Probability</th>
          </tr>
        </thead>
        <tbody>
          {tooth.surfaces.map((surface) => (
            <tr
              key={surface.name}
              className={
                surface.label === 'caries' ? 'bg-danger-50 font-semibold text-danger-600' : ''
              }
            >
              <td className="border-b border-slate-100 py-1.5 capitalize">{surface.name}</td>
              <td className="border-b border-slate-100 py-1.5 capitalize">{surface.label}</td>
              <td className="border-b border-slate-100 py-1.5">
                {(surface.probability * 100).toFixed(0)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
