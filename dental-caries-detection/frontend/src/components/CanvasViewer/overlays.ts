import type { Tooth } from '../../domain/inference';
import { decodeMask } from '../../lib/rle';

const BOX_COLOR = '#2563eb';
const SELECTED_COLOR = '#f97316';
const CARIES_FILL = 'rgba(220, 38, 38, 0.35)';
const SOUND_FILL = 'rgba(37, 99, 235, 0.15)';
const AXIS_COLOR = '#16a34a';

function toothHasCaries(tooth: Tooth): boolean {
  return tooth.surfaces.some((surface) => surface.label === 'caries');
}

export function drawBoundingBox(
  ctx: CanvasRenderingContext2D,
  tooth: Tooth,
  isSelected: boolean
): void {
  const [x, y, w, h] = tooth.bbox;
  ctx.save();
  ctx.strokeStyle = isSelected ? SELECTED_COLOR : BOX_COLOR;
  ctx.lineWidth = isSelected ? 3 : 1.5;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

export function drawMask(
  ctx: CanvasRenderingContext2D,
  tooth: Tooth,
  imgWidth: number,
  imgHeight: number
): void {
  const rings = decodeMask(tooth.mask, imgWidth, imgHeight);
  ctx.save();
  ctx.fillStyle = toothHasCaries(tooth) ? CARIES_FILL : SOUND_FILL;
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

export function drawAxes(ctx: CanvasRenderingContext2D, tooth: Tooth): void {
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

export function drawLabel(ctx: CanvasRenderingContext2D, tooth: Tooth): void {
  const [x, y] = tooth.bbox;
  const text = `FDI ${tooth.fdi}`;
  ctx.save();
  ctx.font = '16px sans-serif';
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#000000';
  ctx.fillStyle = '#ffffff';
  ctx.strokeText(text, x, y - 6);
  ctx.fillText(text, x, y - 6);
  ctx.restore();
}
