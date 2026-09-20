import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { usePolling } from '../usePolling';
import { fetchStatus } from '../../../api/processClient';

vi.mock('../../../api/processClient', () => ({
  fetchStatus: vi.fn(),
}));

const mockedFetchStatus = vi.mocked(fetchStatus);

describe('usePolling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockedFetchStatus.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('polls immediately, then every 2s while active, and stops when active flips false', async () => {
    mockedFetchStatus.mockResolvedValue({ status: 'processing' });
    const onUpdate = vi.fn();

    const { rerender, unmount } = renderHook(({ active }) => usePolling(active, onUpdate), {
      initialProps: { active: true },
    });

    await vi.advanceTimersByTimeAsync(0);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(2000);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(2);
    expect(onUpdate).toHaveBeenCalledWith({ status: 'processing' });

    rerender({ active: false });
    await vi.advanceTimersByTimeAsync(10000);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(2); // no further polling once inactive

    unmount();
  });

  it('never overlaps requests: the next poll is scheduled from settlement, not a fixed clock', async () => {
    let resolveFirst: (v: { status: 'processing' }) => void = () => {};
    mockedFetchStatus.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirst = resolve;
        })
    );
    mockedFetchStatus.mockResolvedValue({ status: 'processing' });

    renderHook(() => usePolling(true, vi.fn()));

    await vi.advanceTimersByTimeAsync(0);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(1);

    // Even if a lot of wall-clock time passes, no 2nd call happens until the
    // 1st settles — setTimeout-recursion, not setInterval.
    await vi.advanceTimersByTimeAsync(10000);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(1);

    resolveFirst({ status: 'processing' });
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(2000);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(2);
  });

  it('backs off exponentially on consecutive failures, capped, and resets on the next success', async () => {
    mockedFetchStatus
      .mockRejectedValueOnce(new Error('network'))
      .mockRejectedValueOnce(new Error('network'))
      .mockResolvedValueOnce({ status: 'processing' })
      .mockResolvedValue({ status: 'processing' });

    const onConnectivityChange = vi.fn();
    renderHook(() => usePolling(true, vi.fn(), onConnectivityChange));

    // Failure #1 -> reconnecting=true, next attempt backs off 2x the base interval.
    await vi.advanceTimersByTimeAsync(0);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(1);
    expect(onConnectivityChange).toHaveBeenLastCalledWith(true);

    await vi.advanceTimersByTimeAsync(3999);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(1); // not yet — still backing off
    await vi.advanceTimersByTimeAsync(1);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(2); // failure #2 fires at 4000ms

    // Failure #2 -> next backoff is 4x the base interval (8000ms).
    await vi.advanceTimersByTimeAsync(7999);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(3); // success

    expect(onConnectivityChange).toHaveBeenLastCalledWith(false);

    // Back to the normal 2s cadence after recovering.
    await vi.advanceTimersByTimeAsync(2000);
    expect(mockedFetchStatus).toHaveBeenCalledTimes(4);
  });

  it('ignores AbortError from an in-flight request cancelled by unmount', async () => {
    mockedFetchStatus.mockImplementation(
      () => new Promise((_, reject) => reject(new DOMException('aborted', 'AbortError')))
    );
    const onUpdate = vi.fn();
    const onConnectivityChange = vi.fn();

    const { unmount } = renderHook(() => usePolling(true, onUpdate, onConnectivityChange));
    await vi.advanceTimersByTimeAsync(0);
    unmount();
    await vi.advanceTimersByTimeAsync(5000);

    expect(onUpdate).not.toHaveBeenCalled();
    expect(onConnectivityChange).not.toHaveBeenCalledWith(true);
  });
});
