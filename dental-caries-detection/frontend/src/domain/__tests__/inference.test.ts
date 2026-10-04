import { describe, expect, it } from 'vitest';
import { parseProcessResponse } from '../inference';
import rawResult from '../../fixtures/ml-result.raw.sample.json';

// The backend forwards the ml-service result JSON as-is inside the `done` body.
const doneResponse = {
  status: 'done',
  image_base64: 'data:image/jpeg;base64,AAAA',
  data: rawResult,
};

describe('parseProcessResponse', () => {
  it('parses the raw ml-service fixture wrapped in a done response', () => {
    expect(() => parseProcessResponse(doneResponse)).not.toThrow();
  });

  it('adapts meta.image_size into data.image and drops the unused meta fields', () => {
    const parsed = parseProcessResponse(doneResponse);
    if (parsed.status !== 'done') throw new Error('expected a done response');

    expect(parsed.data.image).toEqual({ width: 3036, height: 1536 });
    expect(parsed.data.teeth).toHaveLength(4);
    expect('meta' in parsed.data).toBe(false);
  });

  it('keeps optional axes.clamped and surfaces[].method when present', () => {
    const parsed = parseProcessResponse(doneResponse);
    if (parsed.status !== 'done') throw new Error('expected a done response');

    const tooth36 = parsed.data.teeth.find((t) => t.fdi === 36);
    expect(tooth36?.axes.clamped).toBe(false);
    expect(tooth36?.surfaces).toEqual([
      { name: 'occlusal', label: 'caries', probability: 0.815, method: 'RF' },
    ]);
  });

  it('accepts a tooth whose axes have no clamped field', () => {
    const tooth11 = rawResult.teeth.find((t) => t.fdi === 11);
    expect(tooth11?.axes).not.toHaveProperty('clamped');
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

  it('rejects a done payload without meta.image_size', () => {
    const noSize = { ...doneResponse, data: { ...rawResult, meta: { job_id: 1 } } };
    expect(() => parseProcessResponse(noSize)).toThrow();
  });
});
