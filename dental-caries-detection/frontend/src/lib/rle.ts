import type { MaskData } from '../domain/inference';

// A closed ring of image-space [x, y] points. Chosen over Path2D so the same
// data can drive both drawing (CanvasViewer) and pointer hit-testing
// (isPointInPolygon) without needing a live canvas context for the latter.
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

// Standard ray-casting point-in-polygon test, used for canvas hit-testing.
export function isPointInPolygon(point: [number, number], ring: PolygonRing): boolean {
  const [px, py] = point;
  let inside = false;

  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const crossesRay = yi > py !== yj > py;
    if (crossesRay && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }

  return inside;
}
