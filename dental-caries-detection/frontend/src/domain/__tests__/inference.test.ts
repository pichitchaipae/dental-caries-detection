import { describe, expect, it } from 'vitest';
import { parseProcessResponse } from '../inference';
import fixture from '../../fixtures/result.sample.json';

describe('parseProcessResponse', () => {
  it('parses the INT-2 fixture with zero errors', () => {
    expect(() => parseProcessResponse(fixture)).not.toThrow();
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
});
