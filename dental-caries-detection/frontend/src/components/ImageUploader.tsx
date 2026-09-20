import { useCallback, useRef, useState } from 'react';
import { preflightOpg } from '../lib/validation';

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
}

// File picking + pre-flight validation only. Preview and submit now live in
// AnalysisView (FE-5.1), since the thumbnail needs to persist past this
// component's lifetime (it unmounts once a file is selected).
export function ImageUploader({ onFileSelected }: ImageUploaderProps) {
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File | undefined | null) => {
      if (!file) return;
      setError(null);
      void preflightOpg(file).then((result) => {
        if (result.ok) {
          onFileSelected(file);
        } else {
          setError(result.reason);
        }
      });
    },
    [onFileSelected]
  );

  return (
    <div className="flex flex-col gap-3">
      <div
        className={
          'flex flex-col items-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ' +
          (isDragging
            ? 'border-brand-500 bg-brand-50'
            : 'border-slate-300 bg-slate-50 hover:border-brand-400')
        }
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          handleFile(event.dataTransfer.files[0]);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click();
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
          onClick={() => inputRef.current?.click()}
          className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-700"
        >
          Select File
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          className="hidden"
          onChange={(event) => handleFile(event.target.files?.[0])}
        />
      </div>

      {error && (
        <p role="alert" className="text-sm font-medium text-danger-600">
          {error}
        </p>
      )}
    </div>
  );
}
