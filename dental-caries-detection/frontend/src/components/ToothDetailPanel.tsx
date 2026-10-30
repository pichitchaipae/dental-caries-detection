import type { ToothViewModel } from '../features/analysis/analysisTypes';

interface ToothDetailPanelProps {
  tooth: ToothViewModel | null;
  onClose: () => void;
}

export function ToothDetailPanel({ tooth, onClose }: ToothDetailPanelProps) {
  if (!tooth) {
    return (
      <div className="flex h-full min-h-32 items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white/60 p-6 text-center">
        <p className="text-sm text-slate-500">Select a tooth on the image to see its details.</p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-card sm:p-6">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span
            className="h-3 w-3 shrink-0 rounded-full ring-2 ring-white shadow"
            style={{ backgroundColor: tooth.colorKey }}
            aria-hidden="true"
          />
          <h2 className="text-base font-semibold tracking-tight text-slate-900">
            {tooth.displayLabel}
          </h2>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close tooth detail"
          className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
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
      <p className="mb-4 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-600 ring-1 ring-inset ring-slate-200">
        Detection confidence: {(tooth.confidence * 100).toFixed(0)}% · {tooth.cariesSummary}
      </p>
      {tooth.surfaces.length === 0 ? (
        <p className="text-sm text-slate-500">No caries surfaces detected on this tooth.</p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="border-b border-slate-200 px-2 py-2 font-medium">Surface</th>
              <th className="border-b border-slate-200 px-2 py-2 font-medium">Finding</th>
              <th className="border-b border-slate-200 px-2 py-2 font-medium">Probability</th>
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
                <td className="border-b border-slate-100 px-2 py-2 capitalize">{surface.name}</td>
                <td className="border-b border-slate-100 px-2 py-2 capitalize">{surface.label}</td>
                <td className="border-b border-slate-100 px-2 py-2">
                  {surface.probability === null
                    ? '—'
                    : `${(surface.probability * 100).toFixed(0)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
