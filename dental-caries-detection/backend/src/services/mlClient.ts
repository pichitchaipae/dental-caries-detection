/**
 * ML client — HTTP driver for Phase 2 (ML_DRIVER=http).
 *
 * Communicates with the ml-service FastAPI at ML_SERVICE_URL.
 *
 * API contract (JSON):
 *   POST /infer   body: { "jobId": number }  → 202
 *   POST /cancel  body: {}                    → 200
 *   GET  /health                              → 200 | 503
 */

const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? 'http://ml-service:8001';
const FETCH_TIMEOUT_MS = 10_000;

function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs = FETCH_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...init, signal: controller.signal }).finally(() =>
    clearTimeout(timer),
  );
}

/**
 * Dispatch an inference job to the ML service.
 * Calls POST /cancel first to preempt any active job, then POST /infer.
 *
 * @returns true if accepted (202), false on error.
 */
export async function dispatchInference(jobId: number): Promise<boolean> {
  // Preempt any active inference before starting a new one
  await cancelInference().catch(() => {
    /* ignore cancel errors — ml-service may not be running yet */
  });

  let res: Response;
  try {
    res = await fetchWithTimeout(`${ML_SERVICE_URL}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobId }),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[mlClient] POST /infer failed: ${msg}`);
    return false;
  }

  if (res.status === 202) {
    console.info(`[mlClient] job ${jobId} accepted by ml-service`);
    return true;
  }

  const body = await res.text().catch(() => '');
  console.error(
    `[mlClient] POST /infer → HTTP ${res.status}: ${body}`,
  );
  return false;
}

/**
 * Cancel the active inference process on the ML service.
 * Idempotent — safe to call even when no inference is running.
 */
export async function cancelInference(): Promise<void> {
  const res = await fetchWithTimeout(`${ML_SERVICE_URL}/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`POST /cancel → HTTP ${res.status}: ${body}`);
  }
}

/**
 * Check whether the ML service is healthy and all core artifacts are ready.
 */
export async function checkMlHealth(): Promise<{
  ready: boolean;
  artifacts: Record<string, string>;
  active_job_id: number | null;
}> {
  let res: Response;
  try {
    res = await fetchWithTimeout(`${ML_SERVICE_URL}/health`, { method: 'GET' });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[mlClient] GET /health failed: ${msg}`);
    return { ready: false, artifacts: {}, active_job_id: null };
  }

  const json = await res.json().catch(() => ({}));
  return {
    ready: json.ready === true,
    artifacts: json.artifacts ?? {},
    active_job_id: json.active_job_id ?? null,
  };
}
