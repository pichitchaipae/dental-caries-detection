import {
  ABOUT,
  HOW_TO_USE_NOTE,
  HOW_TO_USE_STEPS,
  LICENSE,
  type SiteSectionId,
} from '../data/siteContent';

function AboutContent() {
  const rows: { label: string; value: React.ReactNode }[] = [
    { label: 'Project', value: ABOUT.project },
    {
      label: 'Creators',
      value: (
        <ul className="space-y-1">
          {ABOUT.creators.map((creator) => (
            <li key={creator.name}>
              {creator.name}{' '}
              <span className="ml-1 rounded-full bg-brand-50 px-2 py-0.5 text-xs font-medium tabular-nums text-brand-700 ring-1 ring-inset ring-brand-100">
                {creator.studentId}
              </span>
            </li>
          ))}
        </ul>
      ),
    },
    { label: 'Course', value: ABOUT.course },
    { label: 'Advisor', value: ABOUT.advisor },
    { label: 'Institution', value: ABOUT.institution },
    {
      label: 'Research ethics',
      value: (
        <div className="space-y-1">
          <p className="font-medium">{ABOUT.certification.title}</p>
          <p className="text-slate-600">{ABOUT.certification.detail}</p>
        </div>
      ),
    },
  ];

  return (
    <dl className="divide-y divide-slate-100 text-sm">
      {rows.map((row) => (
        <div
          key={row.label}
          className="grid gap-1 py-3 first:pt-0 last:pb-0 sm:grid-cols-3 sm:gap-4"
        >
          <dt className="font-medium text-slate-500">{row.label}</dt>
          <dd className="text-slate-900 sm:col-span-2">{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function HowToUseContent() {
  return (
    <div className="space-y-5">
      <ol className="space-y-4">
        {HOW_TO_USE_STEPS.map((step, index) => (
          <li key={step.title} className="flex gap-3">
            <span
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-600 text-xs font-semibold text-white shadow-sm shadow-brand-600/30"
              aria-hidden="true"
            >
              {index + 1}
            </span>
            <div>
              <h3 className="text-sm font-semibold text-slate-900">{step.title}</h3>
              <p className="mt-0.5 text-sm leading-relaxed text-slate-600">{step.description}</p>
            </div>
          </li>
        ))}
      </ol>
      <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800 ring-1 ring-inset ring-amber-200">
        {HOW_TO_USE_NOTE}
      </p>
    </div>
  );
}

function LicenseContent() {
  return (
    <div className="space-y-4">
      <div>
        <p className="text-sm font-semibold text-slate-900">{LICENSE.name}</p>
        <p className="text-xs text-slate-500">{LICENSE.notice}</p>
      </div>
      <div className="space-y-3 rounded-xl bg-slate-50 p-4 text-xs leading-relaxed text-slate-600 ring-1 ring-inset ring-slate-200">
        <p className="font-medium text-slate-700">{LICENSE.copyright}</p>
        {LICENSE.body.map((paragraph) => (
          <p key={paragraph.slice(0, 24)}>{paragraph}</p>
        ))}
      </div>
    </div>
  );
}

export function SiteSectionContent({ section }: { section: SiteSectionId }) {
  switch (section) {
    case 'about':
      return <AboutContent />;
    case 'how-to-use':
      return <HowToUseContent />;
    case 'license':
      return <LicenseContent />;
  }
}
