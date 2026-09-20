import { useEffect, useRef } from 'react';
import { fetchStatus } from '../../api/processClient';
import type { ProcessResponse } from '../../domain/inference';

const POLL_INTERVAL_MS = Number(import.meta.env.VITE_POLL_INTERVAL_MS ?? 2000);

// Polls GET /process every POLL_INTERVAL_MS while `active`, per
// project-structure.md Section 6.1 ("about every 2 seconds"). Stops
// immediately (aborting any in-flight request) when `active` goes false —
// AnalysisView flips that on 'done'/'fail'/'Start over'.
export function usePolling(active: boolean, onUpdate: (response: ProcessResponse) => void): void {
  const onUpdateRef = useRef(onUpdate);

  useEffect(() => {
    onUpdateRef.current = onUpdate;
  }, [onUpdate]);

  useEffect(() => {
    if (!active) return;

    let cancelled = false;
    const controller = new AbortController();

    const tick = async () => {
      try {
        const response = await fetchStatus(controller.signal);
        if (!cancelled) onUpdateRef.current(response);
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        // Transient network errors are swallowed here; the next tick retries.
        // Nothing is surfaced to the UI except through onUpdate.
      }
    };

    void tick();
    const intervalId = setInterval(() => void tick(), POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(intervalId);
    };
  }, [active]);
}
