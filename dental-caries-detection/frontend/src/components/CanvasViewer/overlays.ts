import type { ToothViewModel } from '../../features/analysis/analysisTypes';
import { decodeMask } from '../../lib/rle';

const SELECTION_COLOR = '#f97316';
const CARIES_FILL = 'rgba(220, 38, 38, 0.35)';
const SOUND_FILL = 'rgba(37, 99, 235, 0.15)';
const AXIS_COLOR = '#16a34a';
const SELECTION_HALO_PADDING = 6;

// Each primitive draws exactly one tooth, in image-space (the renderer
// applies the world/pan-zoom transform before calling these). FE-6.2's five
// primitives: box, mask, axes, label, and a dedicated selection halo drawn
// as a separate final pass rather than folded into drawBoundingBox.

export function drawBoundingBox(ctx: CanvasRenderingContext2D, tooth: ToothViewModel): void {
  const [x, y, w, h] = tooth.bbox;
  ctx.save();
  ctx.strokeStyle = tooth.colorKey;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
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
    if (ring.length === 0) continue;
    ctx.beginPath();
    ctx.moveTo(ring[0][0], ring[0][1]);
    for (const [px, py] of ring.slice(1)) ctx.lineTo(px, py);
    ctx.closePath();
    ctx.fill();
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
// tooth drawn last with a halo") — a padded, glowing ring around the bbox,
// distinct from that tooth's own identity-colored box.
export function drawSelectionHalo(ctx: CanvasRenderingContext2D, tooth: ToothViewModel): void {
  const [x, y, w, h] = tooth.bbox;
  ctx.save();
  ctx.shadowColor = 'rgba(249, 115, 22, 0.65)';
  ctx.shadowBlur = 12;
  ctx.strokeStyle = SELECTION_COLOR;
  ctx.lineWidth = 3;
  ctx.strokeRect(
    x - SELECTION_HALO_PADDING,
    y - SELECTION_HALO_PADDING,
    w + SELECTION_HALO_PADDING * 2,
    h + SELECTION_HALO_PADDING * 2
  );
  ctx.restore();
}
