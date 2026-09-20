import { describe, expect, it } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { defaultLayerVisibility, useLayerState } from '../layerState';

describe('useLayerState', () => {
  it('starts at the documented defaults', () => {
    const { result } = renderHook(() => useLayerState());
    expect(result.current.layers).toEqual(defaultLayerVisibility);
  });

  it('toggle flips exactly one layer', () => {
    const { result } = renderHook(() => useLayerState());

    act(() => result.current.toggle('axes'));
    expect(result.current.layers).toEqual({ ...defaultLayerVisibility, axes: true });

    act(() => result.current.toggle('boxes'));
    expect(result.current.layers).toEqual({ ...defaultLayerVisibility, axes: true, boxes: false });
  });

  it('setAll sets every layer to the same visibility', () => {
    const { result } = renderHook(() => useLayerState());

    act(() => result.current.setAll(false));
    expect(result.current.layers).toEqual({
      boxes: false,
      masks: false,
      axes: false,
      labels: false,
    });

    act(() => result.current.setAll(true));
    expect(result.current.layers).toEqual({ boxes: true, masks: true, axes: true, labels: true });
  });
});
