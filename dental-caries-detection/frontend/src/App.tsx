import { AnalysisView } from './features/analysis/AnalysisView';

// Single-column layout, no router (project-structure.md Section 4.2: the app
// holds no long-lived state; a refresh fully resets it). The top bar's nav
// links from the design reference (History, About, ...) are intentionally
// NOT rendered as real links here — there is only one view in Phase 1, and a
// nav item that goes nowhere is worse UX than no nav item. See
// z-claude-md/pond-task.md for the full redesign rationale.
function App() {
  const isMocking =
    import.meta.env.DEV && (import.meta.env.VITE_API_MOCKING ?? 'enabled') === 'enabled';

  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2.5">
            <svg
              className="h-6 w-6 text-brand-600"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <path d="M3 12h4l2 5 4-14 2 9h6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div>
              <h1 className="text-lg font-bold leading-tight text-slate-900">
                Dental Caries Surface Classification
              </h1>
              <p className="text-xs text-slate-500">No patient data is stored by this tool.</p>
            </div>
          </div>

          <span
            className={
              'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ' +
              (isMocking ? 'bg-amber-50 text-amber-700' : 'bg-brand-50 text-brand-700')
            }
          >
            <span
              className={
                'h-1.5 w-1.5 rounded-full ' + (isMocking ? 'bg-amber-500' : 'bg-brand-600')
              }
              aria-hidden="true"
            />
            {isMocking ? 'Mock Data Mode' : 'Live Backend'}
          </span>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">
        <AnalysisView />
      </main>

      <footer className="border-t border-slate-200 px-6 py-3 text-xs text-slate-400">
        <div className="mx-auto flex max-w-6xl justify-between">
          <span>v0.1.0</span>
          <span>Port 3000</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
