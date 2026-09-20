import { parseProcessResponse, type ProcessResponse } from '../domain/inference';

const BASE_URL = import.meta.env.VITE_FRONTEND_API_BASE_URL ?? 'http://localhost:8000';

export type ApiErrorKind = 'network' | 'parse' | 'http';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
  }
}

export type SubmitResult = { status: 'processing' } | { status: 'fail'; fail_message: string };

// POST /process — multipart upload, no Content-Type header so the browser
// sets the multipart boundary itself (project-structure.md Section 7.1).
export async function submitOpg(file: File): Promise<SubmitResult> {
  const formData = new FormData();
  formData.append('image', file);

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/process`, { method: 'POST', body: formData });
  } catch {
    throw new ApiError('network', 'Could not reach the server.');
  }

  if (response.status === 202) {
    return { status: 'processing' };
  }

  if (response.status === 415 || response.status === 422 || response.status === 413) {
    let failMessage = 'The image was rejected by the server.';
    try {
      const body = (await response.json()) as { fail_message?: string };
      if (typeof body.fail_message === 'string') failMessage = body.fail_message;
    } catch {
      // Keep the default message if the error body isn't JSON.
    }
    return { status: 'fail', fail_message: failMessage };
  }

  throw new ApiError(
    'http',
    `Unexpected response (${response.status}) from the server.`,
    response.status
  );
}

// GET /process — polled by usePolling every VITE_POLL_INTERVAL_MS.
export async function fetchStatus(signal?: AbortSignal): Promise<ProcessResponse> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/process`, { cache: 'no-store', signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError('network', 'Could not reach the server.');
  }

  if (!response.ok) {
    throw new ApiError(
      'http',
      `Unexpected response (${response.status}) from the server.`,
      response.status
    );
  }

  let json: unknown;
  try {
    json = await response.json();
  } catch {
    throw new ApiError('parse', 'Server response was not valid JSON.');
  }

  try {
    return parseProcessResponse(json);
  } catch {
    throw new ApiError('parse', 'Server response did not match the expected shape.');
  }
}
