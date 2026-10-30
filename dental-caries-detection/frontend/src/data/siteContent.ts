// Static copy for the footer's informational overlays (About / How to Use /
// License). Plain constants on purpose: no backend, no router, no fetching.

export type SiteSectionId = 'about' | 'how-to-use' | 'license';

export const SECTION_TITLES: Record<SiteSectionId, string> = {
  about: 'About',
  'how-to-use': 'How to Use',
  license: 'License',
};

export const ABOUT = {
  project: 'Dental Caries Surface Classification Inference Tool with Machine Learning',
  creators: [
    { name: 'Sukollapat Pisuchpen', studentId: '6687052' },
    { name: 'Pichitchai Paecharoenchai', studentId: '6687033' },
    { name: 'Naris Pholpak', studentId: '6687025' },
  ],
  course: 'ITDS 491-492 Senior Project I & II (Semester 1 & 2, AY 2026)',
  advisor: 'Assistant Professor Dr. Sirawich Vachmanus',
  institution: 'Faculty of ICT, Mahidol University',
  // Research-ethics training held by the creators.
  certification: {
    title: 'Biomedical Responsible Conduct of Research (1 - RCR)',
    detail:
      'The creators have completed the training and hold the certificate from the CITI Program (Collaborative Institutional Training Initiative), University of Miami, with Mahidol University.',
  },
} as const;

export interface HowToUseStep {
  title: string;
  description: string;
}

export const HOW_TO_USE_STEPS: readonly HowToUseStep[] = [
  {
    title: 'Upload a panoramic X-ray',
    description:
      'Drag and drop an OPG image onto the Input Workspace, or click Select File. JPEG or PNG, at least 1000 × 500 px and up to 10 MB.',
  },
  {
    title: 'Run the AI analysis',
    description:
      'Check the preview, then press Run AI Analysis. Processing can take 30 seconds or more; keep the page open until results appear.',
  },
  {
    title: 'Read the image overlays',
    description:
      'Each detected tooth gets a coloured bounding box and its FDI number. Red-filled surfaces are predicted caries; blue-tinted surfaces are predicted sound. Use the Boxes, Masks, Axes and Labels toggles to show or hide layers.',
  },
  {
    title: 'Inspect individual teeth',
    description:
      'Click a tooth on the image or a row in the findings table. The detail panel lists each surface with its finding and probability. Scroll to zoom, drag to pan, and use Fit or 1:1 to reset the view.',
  },
  {
    title: 'Review the findings table',
    description:
      'All detected teeth are listed with detection confidence and any caries surfaces. Click the FDI or Caries surfaces header to sort. Press Start over to analyse another image.',
  },
] as const;

export const HOW_TO_USE_NOTE =
  'This tool is a research prototype and supports, but does not replace, a clinician’s judgement.';

export const LICENSE = {
  name: 'MIT License',
  notice: 'Placeholder text for academic use. Replace with the final license before any release.',
  copyright: 'Copyright (c) 2026 Dental Caries Detection AI project contributors',
  body: [
    'Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:',
    'The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.',
    'THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.',
  ],
} as const;

export const FOOTER = {
  tagline: 'AI-assisted caries detection on panoramic radiographs.',
  institution: ABOUT.institution,
  version: 'v0.1.0',
} as const;
