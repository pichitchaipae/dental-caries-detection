# Frontend Redesign Changelog

## 2026-10-30 — Visual refresh (Tailwind CSS)

**Scope:** `dental-caries-detection/frontend` (React + Vite + Tailwind CSS v4).
Visual-only change: Tailwind classes, layout wrappers, decorative icons and colors.
No React hooks, state, props, event handlers, API calls or text content were changed.
The legacy `caries_surface_classification/app/frontend` client (Tailwind v3) was not touched.

**Verification:** `tsc --noEmit`, `eslint src`, `vitest run` (25/25 tests) and `vite build` all pass.

### Files modified

All paths are relative to `dental-caries-detection/frontend/`.

#### `src/styles/global.css`
- Completed the `brand` color scale (50–900). Components already used `brand-400` and `brand-500`, which were previously undefined and silently produced no styles.
- Added `danger-100/200/700` shades and a `--shadow-card` token (soft two-layer shadow used by all cards).
- Set the `--font-sans` stack (Inter with system fallbacks, no external font download) and enabled antialiasing.
- Added a subtle radial brand-tinted gradient behind the page header area.
- Added a global `:focus-visible` outline so keyboard focus is clearly visible on every control.

#### `src/App.tsx`
- Header is now sticky with a translucent white background and backdrop blur.
- Logo icon sits in a rounded gradient tile (brand-500 → brand-700) with a colored shadow.
- Title/subtitle truncate on narrow screens; responsive paddings (`px-4 sm:px-6`).
- Mode badge gained an inset ring; the "Live Backend" dot now pulses.
- Footer is a translucent bar with a lighter border; main content padding is responsive.

#### `src/features/analysis/AnalysisView.tsx`
- Cards upgraded to `rounded-2xl` with a hairline border, `shadow-card` and responsive padding.
- Section headings ("Input Workspace", "Analysis Results") now have an icon tile beside the title/subtitle.
- Image preview uses a darker frame with a ring; the file name sits in a soft pill.
- Submit error is shown as a tinted alert box with a ring.
- Buttons: primary has a brand shadow and a press effect; disabled "Analyzing…" and secondary "Start over" were restyled to match.
- Empty, processing and failure placeholders use a slate gradient / tinted background with rings and `rounded-xl`; spinner colors adjusted for contrast.

#### `src/components/ImageUploader.tsx`
- Drop zone is taller with a softer background and a hover/drag tint (brand tint + inner shadow while dragging).
- Upload icon is wrapped in a circular badge that changes color while dragging.
- "Select File" button matches the new primary button style.
- Validation error is rendered in a tinted alert row; "Dismiss" link has a clearer hover state.

#### `src/components/ToothDetailPanel.tsx`
- Empty state is a dashed, centered placeholder card; populated state uses the shared card style.
- Color dot has a white ring and shadow; heading has tighter tracking.
- Confidence summary is shown in a soft inset box.
- Surface table: uppercase small-caps style header, more cell padding.
- Close button has a larger, rounded hit area.

#### `src/components/FindingsTable.tsx`
- Wrapper uses the shared card style (`rounded-2xl`, `shadow-card`); caption is larger.
- Header row has a tinted background, uppercase small text and taller cells with hover feedback on sortable columns.
- Selected row shows a brand-colored left accent bar in addition to the tint.
- Rows have more vertical padding; FDI numbers are medium-weight with ringed color dots.
- Caries surfaces are shown as red pill badges instead of plain red text; the empty state has more padding.

#### `src/components/CanvasViewer/CanvasViewer.tsx`
- Layer toggles are now pill-shaped chips; Fit / 1:1 buttons have a white background, shadow and hover border.
- Canvas frame is `rounded-xl` with an inner shadow and ring.
- Brightness/contrast sliders sit in a soft inset panel.
- The "no teeth detected" hint is shown as an amber tinted notice.
- Canvas element classes (`h-full w-full cursor-grab touch-none`) were left unchanged because the renderer depends on them for sizing and interaction.
