import type { InferenceData, SurfaceFinding, Tooth } from '../domain/inference';

// Mimics the API v1 result returned by the real backend: reads the
// real dimensions of the uploaded image and generates plausible synthetic teeth
// inside its bounds. Like the real ml-service, `surfaces` lists only the surfaces
// found with caries — a healthy tooth has an empty array.

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

function buildSurfaces(cariousSurface: SurfaceName | null): SurfaceFinding[] {
  if (cariousSurface === null) return [];
  return [
    {
      name: cariousSurface,
      label: 'caries',
      probability: 0.75 + Math.random() * 0.2,
      method: 'RF',
    },
  ];
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

    return {
      id: i,
      fdi: FDI_CODES[i],
      confidence: 0.9 + i * 0.02,
      bbox,
      mask: { encoding: 'polygon', data: octagonAround(bbox) },
      axes: {
        major: [Math.cos(radians), Math.sin(radians)],
        minor: [-Math.sin(radians), Math.cos(radians)],
        rotation_deg: rotation,
        clamped: false,
      },
      has_caries: i === cariesToothIndex,
      surfaces: buildSurfaces(i === cariesToothIndex ? 'occlusal' : null),
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
