import { describe, expect, it } from 'vitest';
import { parseProcessResponse } from '../inference';
import rawResult from '../../fixtures/ml-result.raw.sample.json';

const apiData = {
  meta: {
    job_id: rawResult.meta.job_id,
    processed_at: new Date(rawResult.meta.completed_at).toISOString(),
    models: { detector: 'unknown', classifier: 'unknown' },
    timings_ms: {},
  },
  image: rawResult.meta.image_size,
  teeth: rawResult.teeth.map((tooth, id) => ({
    ...tooth,
    id,
    axes: { ...tooth.axes, clamped: tooth.axes.clamped ?? false },
    has_caries: tooth.surfaces.length > 0,
    surfaces: tooth.surfaces.map((surface) => ({
      ...surface,
      probability: surface.method === 'RF' ? surface.probability : null,
    })),
  })),
};

const doneResponse = {
  status: 'done',
  image_base64: 'data:image/jpeg;base64,AAAA',
  data: apiData,
};

describe('parseProcessResponse', () => {
  it('parses the API v1 fixture wrapped in a done response', () => {
    expect(() => parseProcessResponse(doneResponse)).not.toThrow();
  });

  it('keeps the API v1 image and metadata fields', () => {
    const parsed = parseProcessResponse(doneResponse);
    if (parsed.status !== 'done') throw new Error('expected a done response');

    expect(parsed.data.image).toEqual({ width: 3036, height: 1536 });
    expect(parsed.data.teeth).toHaveLength(4);
    expect(parsed.data.meta.models).toEqual({ detector: 'unknown', classifier: 'unknown' });
  });

  it('parses axes.clamped and surfaces[].method', () => {
    const parsed = parseProcessResponse(doneResponse);
    if (parsed.status !== 'done') throw new Error('expected a done response');

    const tooth36 = parsed.data.teeth.find((t) => t.fdi === 36);
    expect(tooth36?.axes.clamped).toBe(false);
    expect(tooth36?.surfaces).toEqual([
      { name: 'occlusal', label: 'caries', probability: 0.815, method: 'RF' },
    ]);
  });

  it('requires axes.clamped in API v1', () => {
    const tooth11 = apiData.teeth.find((t) => t.fdi === 11);
    expect(tooth11?.axes).toHaveProperty('clamped');
    expect(() => parseProcessResponse(doneResponse)).not.toThrow();
  });

  it('parses idle, processing, and fail shapes', () => {
    expect(parseProcessResponse({ status: 'idle' })).toEqual({ status: 'idle' });
    expect(parseProcessResponse({ status: 'processing' })).toEqual({ status: 'processing' });
    expect(parseProcessResponse({ status: 'fail', fail_message: 'boom' })).toEqual({
      status: 'fail',
      fail_message: 'boom',
    });
  });

  it('rejects a malformed payload', () => {
    expect(() => parseProcessResponse({ status: 'done' })).toThrow();
  });

  it('rejects a done payload without image', () => {
    const noSize = { ...doneResponse, data: { ...apiData, image: undefined } };
    expect(() => parseProcessResponse(noSize)).toThrow();
  });
});
