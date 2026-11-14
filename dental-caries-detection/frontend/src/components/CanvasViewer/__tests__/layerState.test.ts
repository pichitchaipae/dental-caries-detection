import { describe, expect, it } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { defaultLayerVisibility, useLayerState } from '../layerState';

describe('useLayerState', () => {
  it('starts at the documented defaults', () => {
    const { result } = renderHook(() => useLayerState());
    expect(result.current.layers).toEqual(defaultLayerVisibility);
    expect(defaultLayerVisibility).toEqual({
      masks: true,
      caries: true,
      axes: false,
      labels: true,
    });
  });

  it('has no bounding-box layer any more', () => {
    expect(Object.keys(defaultLayerVisibility)).not.toContain('boxes');
  });

  it('toggle flips exactly one layer', () => {
    const { result } = renderHook(() => useLayerState());

    act(() => result.current.toggle('axes'));
    expect(result.current.layers).toEqual({ ...defaultLayerVisibility, axes: true });

    act(() => result.current.toggle('masks'));
    expect(result.current.layers).toEqual({ ...defaultLayerVisibility, axes: true, masks: false });
  });

  it('toggling caries leaves the tooth masks layer untouched, and back again', () => {
    const { result } = renderHook(() => useLayerState());

    act(() => result.current.toggle('caries'));
    expect(result.current.layers).toEqual({ ...defaultLayerVisibility, caries: false });
    expect(result.current.layers.masks).toBe(true);

    act(() => result.current.toggle('caries'));
    expect(result.current.layers).toEqual(defaultLayerVisibility);
  });

  it('setAll sets every layer to the same visibility', () => {
    const { result } = renderHook(() => useLayerState());

    act(() => result.current.setAll(false));
    expect(result.current.layers).toEqual({
      masks: false,
      caries: false,
      axes: false,
      labels: false,
    });

    act(() => result.current.setAll(true));
    expect(result.current.layers).toEqual({
      masks: true,
      caries: true,
      axes: true,
      labels: true,
    });
  });
});
