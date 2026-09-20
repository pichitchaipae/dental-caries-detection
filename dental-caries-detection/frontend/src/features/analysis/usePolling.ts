import { useEffect, useRef } from 'react';
import { fetchStatus } from '../../api/processClient';
import type { ProcessResponse } from '../../domain/inference';

const POLL_INTERVAL_MS = Number(import.meta.env.VITE_POLL_INTERVAL_MS ?? 2000);
// FE-7.2: capped exponential backoff on transient poll failures, so a
// backend blip doesn't hammer the server, but recovery is still prompt once
// it's back (backoff resets to the base interval on the next success).
const MAX_BACKOFF_MULTIPLIER = 8;

// Polls GET /process every POLL_INTERVAL_MS while `active`, per
// project-structure.md Section 6.1 ("about every 2 seconds"). Stops
// immediately (aborting any in-flight request) when `active` goes false —
// AnalysisView flips that on 'done'/'fail'/'Start over'. Uses recursive
// setTimeout rather than setInterval so a slow/failing request can never
// overlap with the next scheduled tick.
export function usePolling(
  active: boolean,
  onUpdate: (response: ProcessResponse) => void,
  onConnectivityChange?: (isReconnecting: boolean) => void
): void {
  const onUpdateRef = useRef(onUpdate);
  const onConnectivityChangeRef = useRef(onConnectivityChange);

  useEffect(() => {
    onUpdateRef.current = onUpdate;
    onConnectivityChangeRef.current = onConnectivityChange;
  }, [onUpdate, onConnectivityChange]);

  useEffect(() => {
    if (!active) return;

    let cancelled = false;
    let consecutiveFailures = 0;
    let timeoutId: ReturnType<typeof setTimeout>;
    const controller = new AbortController();

    const scheduleNext = (delayMs: number) => {
      timeoutId = setTimeout(() => void tick(), delayMs);
    };

    const tick = async () => {
      try {
        const response = await fetchStatus(controller.signal);
        if (cancelled) return;
        if (consecutiveFailures > 0) {
          consecutiveFailures = 0;
          onConnectivityChangeRef.current?.(false);
        }
        onUpdateRef.current(response);
        scheduleNext(POLL_INTERVAL_MS);
      } catch (err) {
        if (cancelled) return;
        if (err instanceof DOMException && err.name === 'AbortError') return;
        consecutiveFailures += 1;
        onConnectivityChangeRef.current?.(true);
        const backoffMultiplier = Math.min(2 ** consecutiveFailures, MAX_BACKOFF_MULTIPLIER);
        scheduleNext(POLL_INTERVAL_MS * backoffMultiplier);
      }
    };

    void tick();

    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timeoutId);
      onConnectivityChangeRef.current?.(false);
    };
  }, [active]);
}
