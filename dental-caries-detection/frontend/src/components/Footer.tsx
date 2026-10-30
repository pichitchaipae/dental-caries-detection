import { FOOTER, SECTION_TITLES, type SiteSectionId } from '../data/siteContent';

interface FooterProps {
  onOpenSection: (section: SiteSectionId) => void;
}

const FOOTER_LINKS: SiteSectionId[] = ['about', 'how-to-use', 'license'];

export function Footer({ onOpenSection }: FooterProps) {
  return (
    <footer className="mt-8 border-t border-slate-200/70 bg-white/70 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-8 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="flex items-start gap-3">
          <span
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-linear-to-br from-brand-500 to-brand-700 shadow-md shadow-brand-600/25"
            aria-hidden="true"
          >
            <svg
              className="h-4.5 w-4.5 text-white"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M3 12h4l2 5 4-14 2 9h6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <div>
            <p className="text-sm font-semibold text-slate-800">{FOOTER.institution}</p>
            <p className="text-xs text-slate-500">{FOOTER.tagline}</p>
          </div>
        </div>

        <nav aria-label="Site information" className="flex flex-wrap items-center gap-2">
          {FOOTER_LINKS.map((section) => (
            <button
              key={section}
              type="button"
              aria-haspopup="dialog"
              onClick={() => onOpenSection(section)}
              className="rounded-full border border-slate-200 bg-white px-4 py-1.5 text-sm font-medium text-slate-600 shadow-sm transition hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700"
            >
              {SECTION_TITLES[section]}
            </button>
          ))}
        </nav>
      </div>

      <div className="border-t border-slate-100">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 text-xs text-slate-400 sm:px-6">
          <span>No patient data is stored by this tool.</span>
          <span>{FOOTER.version}</span>
        </div>
      </div>
    </footer>
  );
}
