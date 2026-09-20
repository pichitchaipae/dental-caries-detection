import type { MaskData } from '../domain/inference';

// A closed ring of image-space [x, y] points, fed to Path2D by
// CanvasViewer/overlays.ts for both drawing and ctx.isPointInPath hit-testing.
export type PolygonRing = [number, number][];

export function decodeMask(mask: MaskData, _imgWidth: number, _imgHeight: number): PolygonRing[] {
  if (mask.encoding === 'polygon') {
    return [mask.data as PolygonRing];
  }

  // The Phase 1 Mock ML driver only ever emits `encoding: 'polygon'`
  // (docs-md/task-pm-phase1.md INT-2). RLE decoding is real Phase 2 ML
  // service work; this stub keeps the type/signature in place until then.
  throw new Error('RLE mask decoding is not implemented in Phase 1.');
}
