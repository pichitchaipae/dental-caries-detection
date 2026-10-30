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
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-slate-200/70 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-linear-to-br from-brand-500 to-brand-700 shadow-md shadow-brand-600/25">
              <svg
                className="h-5 w-5 text-white"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                aria-hidden="true"
              >
                <path d="M3 12h4l2 5 4-14 2 9h6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold leading-tight tracking-tight text-slate-900 sm:text-lg">
                Dental Caries Surface Classification
              </h1>
              <p className="truncate text-xs text-slate-500">
                No patient data is stored by this tool.
              </p>
            </div>
          </div>

          <span
            className={
              'inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ring-1 ring-inset ' +
              (isMocking
                ? 'bg-amber-50 text-amber-700 ring-amber-200'
                : 'bg-brand-50 text-brand-700 ring-brand-200')
            }
          >
            <span
              className={
                'h-1.5 w-1.5 rounded-full ' +
                (isMocking ? 'bg-amber-500' : 'animate-pulse bg-brand-600')
              }
              aria-hidden="true"
            />
            {isMocking ? 'Mock Data Mode' : 'Live Backend'}
          </span>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
        <AnalysisView />
      </main>

      <footer className="border-t border-slate-200/70 bg-white/60 px-4 py-4 text-xs text-slate-400 sm:px-6">
        <div className="mx-auto flex max-w-6xl justify-between">
          <span>v0.1.0</span>
          <span>Port 3000</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
