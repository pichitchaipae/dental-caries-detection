import { useCallback, useEffect, useMemo, useState } from 'react';
import { ApiError, submitOpg } from '../../api/processClient';
import type { ProcessResponse } from '../../domain/inference';
import { ImageUploader } from '../../components/ImageUploader';
import { CanvasViewer } from '../../components/CanvasViewer/CanvasViewer';
import { ToothDetailPanel } from '../../components/ToothDetailPanel';
import { FindingsTable } from '../../components/FindingsTable';
import { usePolling } from './usePolling';
import { toViewModel } from './analysisTypes';
import type { WorkflowState } from './analysisTypes';

// Owns the whole upload -> processing -> results workflow. No router, no
// external store (FE-4.1: local state + hooks only; a refresh resets
// everything). `preview` is tracked separately from `state` so the left
// "Input Workspace" panel can keep showing the current image's thumbnail
// through processing/done/fail, not just while it's still unsubmitted.
export function AnalysisView() {
  const [state, setState] = useState<WorkflowState>({ phase: 'empty' });
  const [preview, setPreview] = useState<{ url: string; name: string } | null>(null);
  const [selectedToothId, setSelectedToothId] = useState<number | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isReconnecting, setIsReconnecting] = useState(false);

  useEffect(() => {
    if (!preview) return;
    return () => URL.revokeObjectURL(preview.url);
  }, [preview]);

  const handleFileSelected = useCallback((file: File) => {
    setSelectedToothId(null);
    setSubmitError(null);
    setPreview({ url: URL.createObjectURL(file), name: file.name });
    setState({ phase: 'selected', file });
  }, []);

  const handleSubmit = useCallback(() => {
    if (state.phase !== 'selected') return;
    const { file } = state;
    setSubmitError(null);
    setState({ phase: 'submitting', file });

    void (async () => {
      try {
        const result = await submitOpg(file);
        if (result.status === 'processing') {
          setState({ phase: 'processing' });
        } else {
          setState({ phase: 'fail', failMessage: result.fail_message });
        }
      } catch (err) {
        // FE-7.2: a network error on submit keeps the file selected so the
        // clinician can just retry, instead of losing their upload.
        const message =
          err instanceof ApiError ? err.message : 'Could not reach the server. Please try again.';
        setSubmitError(message);
        setState({ phase: 'selected', file });
      }
    })();
  }, [state]);

  const handlePollUpdate = useCallback((response: ProcessResponse) => {
    if (response.status === 'done') {
      setState({ phase: 'done', imageBase64: response.image_base64, data: response.data });
    } else if (response.status === 'fail') {
      setState({ phase: 'fail', failMessage: response.fail_message });
    }
    // 'idle' / 'processing' responses leave the current 'processing' phase as-is.
  }, []);

  const handleStartOver = useCallback(() => {
    setSelectedToothId(null);
    setSubmitError(null);
    setIsReconnecting(false);
    setPreview(null);
    setState({ phase: 'empty' });
  }, []);

  usePolling(state.phase === 'processing', handlePollUpdate, setIsReconnecting);

  const isBusy = state.phase === 'submitting' || state.phase === 'processing';

  // FE-5.4: the view-model layer — CanvasViewer / ToothDetailPanel /
  // FindingsTable all consume ToothViewModel, never the raw InferenceData.
  const viewModel = useMemo(
    () => (state.phase === 'done' ? toViewModel(state.data) : null),
    [state]
  );
  const selectedTooth = viewModel?.teeth.find((tooth) => tooth.id === selectedToothId) ?? null;

  return (
    <div className="flex flex-col gap-8">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left: Input Workspace */}
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="text-base font-semibold text-slate-900">Input Workspace</h2>
          <p className="mb-4 text-sm text-slate-500">Upload a panoramic X-ray to begin analysis.</p>

          {preview ? (
            <div className="flex flex-col gap-4">
              <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-900">
                <img
                  src={preview.url}
                  alt="Selected OPG preview"
                  className="max-h-64 w-full object-contain"
                />
              </div>
              <p className="truncate text-sm text-slate-600">{preview.name}</p>

              {submitError && (
                <p role="alert" className="text-sm font-medium text-danger-600">
                  {submitError}
                </p>
              )}

              {state.phase === 'selected' && (
                <button
                  type="button"
                  onClick={handleSubmit}
                  className="rounded-lg bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-brand-700"
                >
                  Run AI Analysis
                </button>
              )}
              {isBusy && (
                <button
                  type="button"
                  disabled
                  className="cursor-not-allowed rounded-lg bg-slate-100 px-4 py-2.5 text-sm font-semibold text-slate-400"
                >
                  Analyzing…
                </button>
              )}
              {(state.phase === 'done' || state.phase === 'fail') && (
                <button
                  type="button"
                  onClick={handleStartOver}
                  className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                >
                  Start over
                </button>
              )}
            </div>
          ) : (
            <ImageUploader onFileSelected={handleFileSelected} />
          )}
        </section>

        {/* Right: Analysis Results */}
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="text-base font-semibold text-slate-900">Analysis Results</h2>
          <p className="mb-4 text-sm text-slate-500">AI detected findings and confidence scores.</p>

          {state.phase === 'empty' || state.phase === 'selected' ? (
            <div className="flex aspect-video items-center justify-center rounded-lg bg-slate-900 text-slate-400">
              <div className="flex flex-col items-center gap-2 text-sm">
                <svg
                  className="h-8 w-8"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path
                    d="M9 12h6m-6 4h6M9 8h6M5 4h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                Awaiting analysis…
              </div>
            </div>
          ) : isBusy ? (
            <div
              className="flex aspect-video flex-col items-center justify-center gap-3 rounded-lg bg-slate-900 text-slate-300"
              role="status"
            >
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-slate-600 border-t-brand-500" />
              <p className="max-w-xs px-4 text-center text-sm">
                Analyzing radiograph — this can take 30 seconds or more.
              </p>
              {isReconnecting && (
                <p
                  role="status"
                  className="rounded-full bg-amber-500/20 px-3 py-1 text-xs font-medium text-amber-300"
                >
                  Reconnecting to server…
                </p>
              )}
            </div>
          ) : state.phase === 'fail' ? (
            <div
              className="flex aspect-video flex-col items-center justify-center gap-3 rounded-lg bg-danger-50 px-6 text-center"
              role="alert"
            >
              <p className="font-medium text-danger-600">{state.failMessage}</p>
            </div>
          ) : viewModel ? (
            <CanvasViewer
              imageBase64={state.imageBase64}
              imageSize={viewModel.image}
              teeth={viewModel.teeth}
              selectedToothId={selectedToothId}
              onSelectTooth={setSelectedToothId}
            />
          ) : null}
        </section>
      </div>

      {state.phase === 'done' && viewModel && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-1">
            <ToothDetailPanel tooth={selectedTooth} />
          </div>
          <div className="lg:col-span-2">
            <FindingsTable
              teeth={viewModel.teeth}
              selectedToothId={selectedToothId}
              onSelectTooth={setSelectedToothId}
            />
          </div>
        </div>
      )}
    </div>
  );
}
