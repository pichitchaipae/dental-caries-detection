import type { ToothViewModel } from '../features/analysis/analysisTypes';

interface ToothDetailPanelProps {
  tooth: ToothViewModel | null;
  onClose: () => void;
}

export function ToothDetailPanel({ tooth, onClose }: ToothDetailPanelProps) {
  if (!tooth) {
    return (
      <div className="flex h-full items-center rounded-xl border border-slate-200 bg-white p-6">
        <p className="text-sm text-slate-500">Select a tooth on the image to see its details.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="h-2.5 w-2.5 shrink-0 rounded-full"
            style={{ backgroundColor: tooth.colorKey }}
            aria-hidden="true"
          />
          <h2 className="text-base font-semibold text-slate-900">{tooth.displayLabel}</h2>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close tooth detail"
          className="rounded p-1 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
        >
          <svg
            className="h-4 w-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M6 6l12 12M18 6 6 18" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
      <p className="mb-3 text-sm text-slate-500">
        Detection confidence: {(tooth.confidence * 100).toFixed(0)}% · {tooth.cariesSummary}{' '}
        affected
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
