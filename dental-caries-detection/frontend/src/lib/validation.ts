// Pre-flight only: gives the clinician fast, friendly feedback. The backend
// (backend/src/lib/validation.ts, BE-3.1) remains authoritative — these
// defaults are kept in sync with its env-driven constraints by convention,
// not by a shared import (the two apps don't share a module boundary).

export interface OpgConstraints {
  maxBytes: number;
  minWidth: number;
  minHeight: number;
  acceptedTypes: string[];
}

const DEFAULT_MAX_IMAGE_MB = 25;
const DEFAULT_MIN_WIDTH = 1000;
const DEFAULT_MIN_HEIGHT = 500;

function envNumber(value: string | undefined, fallback: number): number {
  const parsed = value === undefined ? NaN : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export const opgConstraints: OpgConstraints = {
  maxBytes: envNumber(import.meta.env.VITE_MAX_IMAGE_MB, DEFAULT_MAX_IMAGE_MB) * 1024 * 1024,
  minWidth: envNumber(import.meta.env.VITE_MIN_IMAGE_WIDTH, DEFAULT_MIN_WIDTH),
  minHeight: envNumber(import.meta.env.VITE_MIN_IMAGE_HEIGHT, DEFAULT_MIN_HEIGHT),
  acceptedTypes: ['image/jpeg', 'image/png'],
};

export type PreflightResult =
  { ok: true; width: number; height: number } | { ok: false; reason: string };

export async function preflightOpg(
  file: File,
  constraints: OpgConstraints = opgConstraints
): Promise<PreflightResult> {
  if (!constraints.acceptedTypes.includes(file.type)) {
    return { ok: false, reason: 'Only JPEG or PNG OPG images are accepted.' };
  }

  if (file.size > constraints.maxBytes) {
    const maxMb = Math.round(constraints.maxBytes / (1024 * 1024));
    return { ok: false, reason: `Image is too large. Maximum size is ${maxMb} MB.` };
  }

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return { ok: false, reason: 'File could not be read as an image.' };
  }

  const { width, height } = bitmap;
  bitmap.close();

  if (width < constraints.minWidth || height < constraints.minHeight) {
    return {
      ok: false,
      reason: `Image resolution too low for OPG analysis (minimum ${constraints.minWidth}x${constraints.minHeight}px).`,
    };
  }

  return { ok: true, width, height };
}
