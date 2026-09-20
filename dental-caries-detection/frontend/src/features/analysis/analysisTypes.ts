import type { InferenceData } from '../../domain/inference';

// `file`/`previewUrl` for the selected image live in AnalysisView's separate
// `preview` state (not here), since the left "Input Workspace" panel needs to
// keep showing the thumbnail through 'processing'/'done'/'fail' too, not just
// while the phase is still 'selected'.
export type WorkflowState =
  | { phase: 'empty' }
  | { phase: 'selected'; file: File }
  | { phase: 'submitting'; file: File }
  | { phase: 'processing' }
  | { phase: 'done'; imageBase64: string; data: InferenceData }
  | { phase: 'fail'; failMessage: string };
