import { describe, expect, it } from 'vitest';
import { parseProcessResponse } from '../inference';
import { loadV2Data, v2DoneResponse } from '../../test-utils/v2Fixture';

// Deep-cloned JSON so each test can mutate its own copy of the fixture.
function cloneDone(): {
  status: string;
  image_base64: string;
  data: ReturnType<typeof loadV2Data> & Record<string, unknown>;
} {
  return structuredClone(v2DoneResponse) as ReturnType<typeof cloneDone>;
}

describe('parseProcessResponse (API v2 draft)', () => {
  it('parses the backend v2 fixture wrapped in a done response', () => {
    expect(() => parseProcessResponse(v2DoneResponse)).not.toThrow();
  });

  it('keeps the image and metadata fields', () => {
    const data = loadV2Data();

    expect(data.image).toEqual({ width: 3037, height: 1536 });
    expect(data.teeth).toHaveLength(29);
    expect(data.meta.models.detector).toBe('Tooth_seg_pano_20250319.pt');
  });

  it('parses caries[] with id, surface, probability, mask and centroid', () => {
    const data = loadV2Data();
    const tooth45 = data.teeth.find((t) => t.fdi === 45);

    expect(tooth45?.caries).toHaveLength(1);
    const lesion = tooth45!.caries[0];
    expect(Object.keys(lesion).sort()).toEqual(
      ['centroid', 'id', 'mask', 'probability', 'surface'].sort()
    );
    expect(lesion.id).toBe(0);
    expect(lesion.surface).toBe('distal');
    expect(lesion.probability).toBe(0.3152);
    expect(lesion.mask.encoding).toBe('polygon');
    expect(lesion.mask.data).toHaveLength(20);
    expect(lesion.centroid).toEqual([1166.76, 930.32]);
  });

  it('parses a sound tooth with caries: []', () => {
    const data = loadV2Data();
    const sound = data.teeth.filter((t) => t.caries.length === 0);

    expect(sound).toHaveLength(27);
    expect(sound.every((t) => t.surfaces.length === 0 && !t.has_caries)).toBe(true);
  });

  it('keeps the tooth bbox but drops tooth confidence and lesion bbox/confidence', () => {
    // Inject the removed v1 fields: the schema must strip them, not expose them.
    const done = cloneDone();
    const tooth = done.data.teeth.find((t) => t.caries.length > 0)!;
    Object.assign(tooth, { confidence: 0.99 });
    Object.assign(tooth.caries[0], { bbox: [1, 2, 3, 4], confidence: 0.5 });

    const parsed = parseProcessResponse(done);
    if (parsed.status !== 'done') throw new Error('expected a done response');
    const parsedTooth = parsed.data.teeth.find((t) => t.caries.length > 0)!;

    expect(parsedTooth.bbox).toHaveLength(4);
    expect(parsedTooth).not.toHaveProperty('confidence');
    expect(parsedTooth.caries[0]).not.toHaveProperty('bbox');
    expect(parsedTooth.caries[0]).not.toHaveProperty('confidence');
  });

  it('accepts a lesion whose surface is null', () => {
    const done = cloneDone();
    done.data.teeth[0].caries[0].surface = null;

    const parsed = parseProcessResponse(done);
    if (parsed.status !== 'done') throw new Error('expected a done response');
    expect(parsed.data.teeth[0].caries[0].surface).toBeNull();
  });

  it('rejects a tooth without caries[] (required in v2)', () => {
    const done = cloneDone();
    delete (done.data.teeth[0] as { caries?: unknown }).caries;

    expect(() => parseProcessResponse(done)).toThrow();
  });

  it('rejects a v1 lesion that still uses confidence instead of probability', () => {
    const done = cloneDone();
    const lesion = done.data.teeth[0].caries[0] as unknown as Record<string, unknown>;
    lesion.confidence = lesion.probability;
    delete lesion.probability;

    expect(() => parseProcessResponse(done)).toThrow();
  });

  it('rejects an unknown surface name and a probability outside 0-1', () => {
    const badSurface = cloneDone();
    (badSurface.data.teeth[0].caries[0] as { surface: string }).surface = 'buccal';
    expect(() => parseProcessResponse(badSurface)).toThrow();

    const badProbability = cloneDone();
    badProbability.data.teeth[0].caries[0].probability = 1.2;
    expect(() => parseProcessResponse(badProbability)).toThrow();
  });

  it.each([0, 1, 2])(
    'still parses the whole result when a lesion mask has only %i point(s)',
    (pointCount) => {
      const done = cloneDone();
      const lesion = done.data.teeth[0].caries[0];
      lesion.mask.data = lesion.mask.data.slice(0, pointCount);

      const parsed = parseProcessResponse(done);
      if (parsed.status !== 'done') throw new Error('expected a done response');

      expect(parsed.data.teeth).toHaveLength(29);
      expect(parsed.data.teeth[0].caries[0].mask.data).toHaveLength(pointCount);
      // The other lesion (FDI 26) is untouched.
      expect(parsed.data.teeth.find((t) => t.fdi === 26)?.caries[0].mask.data).toHaveLength(59);
    }
  );

  it('still requires lesion mask points to be [x, y] pairs', () => {
    const done = cloneDone();
    done.data.teeth[0].caries[0].mask.data = [[1]] as unknown as [number, number][];
    expect(() => parseProcessResponse(done)).toThrow();
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
    const done = cloneDone();
    delete (done.data as { image?: unknown }).image;
    expect(() => parseProcessResponse(done)).toThrow();
  });
});
