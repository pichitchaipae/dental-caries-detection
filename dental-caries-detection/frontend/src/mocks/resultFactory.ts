import type { CariesLesion, InferenceData, SurfaceFinding, Tooth } from '../domain/inference';

// Mimics the API v2 draft result returned by the real backend
// (docs-md/api-contract-v2-draft.md): reads the real dimensions of the
// uploaded image and generates plausible synthetic teeth inside its bounds.
// Like the real ml-service, `surfaces` lists only the surfaces found with
// caries — a healthy tooth has an empty array — and `caries` lists the lesion
// masks (always present; [] for a sound tooth). There is no tooth
// `confidence` or lesion `bbox` any more.

const FDI_CODES = [16, 36, 46] as const;
type SurfaceName = SurfaceFinding['name'];

function fileToDataUri(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Could not read the uploaded file.'));
    reader.readAsDataURL(file);
  });
}

// A rectangle bbox softened into an octagon, per BE-3.4's documented mask
// shape ("a simple polygon mask (rectangle or octagon around the bbox)").
function octagonAround([x, y, w, h]: [number, number, number, number]): [number, number][] {
  const cut = Math.min(w, h) * 0.25;
  return [
    [x + cut, y],
    [x + w - cut, y],
    [x + w, y + cut],
    [x + w, y + h - cut],
    [x + w - cut, y + h],
    [x + cut, y + h],
    [x, y + h - cut],
    [x, y + cut],
  ];
}

// An 8-point blob (ellipse) standing in for a lesion polygon.
function blobAt(cx: number, cy: number, rx: number, ry: number): [number, number][] {
  return Array.from({ length: 8 }, (_, k) => {
    const angle = (k * Math.PI) / 4;
    return [Math.round(cx + rx * Math.cos(angle)), Math.round(cy + ry * Math.sin(angle))];
  });
}

function centroidOf(points: [number, number][]): [number, number] {
  const sum = points.reduce(([sx, sy], [px, py]) => [sx + px, sy + py], [0, 0]);
  return [Math.round(sum[0] / points.length), Math.round(sum[1] / points.length)];
}

// Two lesions on the flagged tooth: one assigned to a surface and one with
// `surface: null`, the state the real backend emits until ML-10 lands, so the
// "Unspecified" UI path is exercised on every mock run. The surface summary's
// probability is the max of the lesions assigned to it (API v2 draft §2.1).
function buildCaries(
  [x, y, w, h]: [number, number, number, number],
  cariousSurface: SurfaceName
): { surfaces: SurfaceFinding[]; caries: CariesLesion[] } {
  const surfaceProbability = 0.75 + Math.random() * 0.2;
  const first = blobAt(x + w * 0.5, y + h * 0.3, w * 0.18, h * 0.08);
  const second = blobAt(x + w * 0.35, y + h * 0.7, w * 0.12, h * 0.06);
  return {
    surfaces: [
      {
        name: cariousSurface,
        label: 'caries',
        probability: surfaceProbability,
        method: 'Geometry_Heuristic',
      },
    ],
    caries: [
      {
        id: 0,
        surface: cariousSurface,
        probability: surfaceProbability,
        mask: { encoding: 'polygon', data: first },
        centroid: centroidOf(first),
      },
      {
        id: 1,
        surface: null,
        probability: Math.min(0.7, surfaceProbability - 0.1),
        mask: { encoding: 'polygon', data: second },
        centroid: centroidOf(second),
      },
    ],
  };
}

function generateSyntheticTeeth(imgWidth: number, imgHeight: number): Tooth[] {
  const count = FDI_CODES.length;
  const archTop = imgHeight * 0.35;
  const archBandHeight = imgHeight * 0.3;
  const toothWidth = imgWidth * 0.08;
  const toothHeight = archBandHeight * 0.8;
  const spacing = (imgWidth * 0.7) / count;
  const startX = imgWidth * 0.15;

  // One tooth is deliberately flagged with a caries surface so the UI's
  // caries highlighting has something real to demonstrate on every run.
  const cariesToothIndex = 1;

  return Array.from({ length: count }, (_, i) => {
    const x = Math.round(startX + i * spacing);
    const y = Math.round(archTop + (i % 2 === 0 ? 0 : archBandHeight * 0.15));
    const bbox: [number, number, number, number] = [
      x,
      y,
      Math.round(toothWidth),
      Math.round(toothHeight),
    ];
    const rotation = (i - (count - 1) / 2) * 8;
    const radians = (rotation * Math.PI) / 180;

    const findings =
      i === cariesToothIndex ? buildCaries(bbox, 'occlusal') : { surfaces: [], caries: [] };

    return {
      id: i,
      fdi: FDI_CODES[i],
      bbox,
      mask: { encoding: 'polygon', data: octagonAround(bbox) },
      axes: {
        major: [Math.cos(radians), Math.sin(radians)],
        minor: [-Math.sin(radians), Math.cos(radians)],
        rotation_deg: rotation,
        clamped: false,
      },
      has_caries: findings.surfaces.length > 0,
      surfaces: findings.surfaces,
      caries: findings.caries,
    } satisfies Tooth;
  });
}

export async function buildMockResult(
  file: File
): Promise<{ data: InferenceData; imageBase64: string }> {
  const [imageBase64, bitmap] = await Promise.all([fileToDataUri(file), createImageBitmap(file)]);
  const { width, height } = bitmap;
  bitmap.close();

  const teeth = generateSyntheticTeeth(width, height);

  return {
    imageBase64,
    data: {
      meta: {
        job_id: Date.now(),
        processed_at: new Date().toISOString(),
        models: { detector: 'mock', classifier: 'mock' },
        timings_ms: {},
      },
      image: { width, height },
      teeth,
    },
  };
}
