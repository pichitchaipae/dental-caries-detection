import type { ToothViewModel } from '../../features/analysis/analysisTypes';
import { decodeMask } from '../../lib/rle';

const SELECTION_COLOR = '#f97316';
const CARIES_FILL = 'rgba(220, 38, 38, 0.35)';
const SOUND_FILL = 'rgba(37, 99, 235, 0.15)';
// Lesion layer: amber, deliberately not red so it still reads on top of the
// red fill of a carious tooth. The opaque stroke keeps small lesions visible
// at fit-to-view zoom.
export const CARIES_MASK_FILL = 'rgba(250, 204, 21, 0.6)';
export const CARIES_MASK_STROKE = '#fef08a';
const CARIES_MASK_LINE_WIDTH = 3; // image-space px
const MIN_POLYGON_POINTS = 3;
const AXIS_COLOR = '#16a34a';

// Each primitive draws exactly one tooth, in image-space (the renderer
// applies the world/pan-zoom transform before calling these). Primitives:
// tooth mask, caries mask, axes, label, and a dedicated selection halo drawn
// as a separate final pass. There is no bounding-box primitive: API v2 draws
// masks only (docs-md/api-contract-v2-draft.md §0), so a tooth's bbox is
// used for label placement and hit-testing but never rendered.

function tracePolygon(ctx: CanvasRenderingContext2D, ring: [number, number][]): boolean {
  if (ring.length === 0) return false;
  ctx.beginPath();
  ctx.moveTo(ring[0][0], ring[0][1]);
  for (const [px, py] of ring.slice(1)) ctx.lineTo(px, py);
  ctx.closePath();
  return true;
}

export function drawMask(
  ctx: CanvasRenderingContext2D,
  tooth: ToothViewModel,
  imgWidth: number,
  imgHeight: number
): void {
  const rings = decodeMask(tooth.mask, imgWidth, imgHeight);
  ctx.save();
  ctx.fillStyle = tooth.hasCaries ? CARIES_FILL : SOUND_FILL;
  for (const ring of rings) {
    if (tracePolygon(ctx, ring)) ctx.fill();
  }
  ctx.restore();
}

// Fills each lesion polygon of one tooth (pano pixel coordinates, same space
// as the tooth mask, so no extra transform). Drawn after the tooth masks so
// the lesion sits on top of the tooth fill.
export function drawCariesMasks(ctx: CanvasRenderingContext2D, tooth: ToothViewModel): void {
  if (tooth.caries.length === 0) return;
  ctx.save();
  ctx.fillStyle = CARIES_MASK_FILL;
  ctx.strokeStyle = CARIES_MASK_STROKE;
  ctx.lineWidth = CARIES_MASK_LINE_WIDTH;
  ctx.lineJoin = 'round';
  for (const lesion of tooth.caries) {
    // A lesion needs at least 3 points to enclose an area; the wire schema
    // lets shorter ones through so they are skipped here, not fatal.
    if (lesion.mask.data.length < MIN_POLYGON_POINTS) continue;
    if (tracePolygon(ctx, lesion.mask.data)) {
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();
}

export function drawAxes(ctx: CanvasRenderingContext2D, tooth: ToothViewModel): void {
  const [x, y, w, h] = tooth.bbox;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const scale = Math.max(w, h) * 0.6;
  const { major, minor } = tooth.axes;

  ctx.save();
  ctx.strokeStyle = AXIS_COLOR;
  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.moveTo(cx - major[0] * scale, cy - major[1] * scale);
  ctx.lineTo(cx + major[0] * scale, cy + major[1] * scale);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx - minor[0] * scale * 0.6, cy - minor[1] * scale * 0.6);
  ctx.lineTo(cx + minor[0] * scale * 0.6, cy + minor[1] * scale * 0.6);
  ctx.stroke();

  ctx.restore();
}

export function drawLabel(ctx: CanvasRenderingContext2D, tooth: ToothViewModel): void {
  const [x, y] = tooth.bbox;
  ctx.save();
  ctx.font = '16px sans-serif';
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#000000';
  ctx.fillStyle = '#ffffff';
  ctx.strokeText(tooth.displayLabel, x, y - 6);
  ctx.fillText(tooth.displayLabel, x, y - 6);
  ctx.restore();
}

// FE-6.2: dedicated emphasis primitive for the selected tooth, drawn as the
// final pass over everything else (project-structure.md 10.1: "Selected
// tooth drawn last with a halo"). It outlines the tooth's own mask with a
// glow instead of a padded rectangle, so selecting a tooth never brings a
// bounding box back on screen.
export function drawSelectionHalo(
  ctx: CanvasRenderingContext2D,
  tooth: ToothViewModel,
  imgWidth: number,
  imgHeight: number
): void {
  const rings = decodeMask(tooth.mask, imgWidth, imgHeight);
  ctx.save();
  ctx.shadowColor = 'rgba(249, 115, 22, 0.65)';
  ctx.shadowBlur = 12;
  ctx.strokeStyle = SELECTION_COLOR;
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  for (const ring of rings) {
    if (tracePolygon(ctx, ring)) ctx.stroke();
  }
  ctx.restore();
}
