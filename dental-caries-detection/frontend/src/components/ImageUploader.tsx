import { useCallback, useRef, useState } from 'react';
import { preflightOpg } from '../lib/validation';

interface ImageUploaderProps {
  onValidFile: (file: File, meta: { width: number; height: number }) => void;
  // Fires when the clinician dismisses a pre-flight validation error. This
  // component doesn't hold a "selected" state of its own (AnalysisView owns
  // that, so the thumbnail persists past processing/done/fail) — `onClear`
  // exists for the parent to react to, e.g. clearing an unrelated stale error.
  onClear: () => void;
  disabled?: boolean;
}

export function ImageUploader({ onValidFile, onClear, disabled = false }: ImageUploaderProps) {
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File | undefined | null) => {
      if (!file || disabled) return;
      setError(null);
      void preflightOpg(file).then((result) => {
        if (result.ok) {
          onValidFile(file, { width: result.width, height: result.height });
        } else {
          setError(result.reason);
        }
      });
    },
    [onValidFile, disabled]
  );

  const handleDismissError = useCallback(() => {
    setError(null);
    onClear();
  }, [onClear]);

  return (
    <div className="flex flex-col gap-3">
      <div
        aria-disabled={disabled}
        className={
          'flex flex-col items-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ' +
          (disabled
            ? 'cursor-not-allowed border-slate-200 bg-slate-50 opacity-60'
            : isDragging
              ? 'cursor-pointer border-brand-500 bg-brand-50'
              : 'cursor-pointer border-slate-300 bg-slate-50 hover:border-brand-400')
        }
        onDragOver={(event) => {
          if (disabled) return;
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (disabled) return;
          handleFile(event.dataTransfer.files[0]);
        }}
        onClick={() => {
          if (!disabled) inputRef.current?.click();
        }}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(event) => {
          if (!disabled && (event.key === 'Enter' || event.key === ' ')) inputRef.current?.click();
        }}
      >
        <svg
          className="h-10 w-10 text-slate-400"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          aria-hidden="true"
        >
          <path
            d="M12 16V4m0 0 4 4m-4-4-4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <div>
          <p className="font-medium text-slate-700">Drag & drop an OPG image</p>
          <p className="text-sm text-slate-500">or click to browse (JPEG, PNG)</p>
        </div>
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Select File
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          disabled={disabled}
          className="hidden"
          onChange={(event) => handleFile(event.target.files?.[0])}
        />
      </div>

      {error && (
        <div className="flex items-center justify-between gap-2">
          <p role="alert" className="text-sm font-medium text-danger-600">
            {error}
          </p>
          <button
            type="button"
            onClick={handleDismissError}
            className="shrink-0 text-xs font-medium text-slate-400 hover:text-slate-600"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
