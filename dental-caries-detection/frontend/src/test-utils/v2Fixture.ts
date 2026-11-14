import { parseProcessResponse, type InferenceData } from '../domain/inference';
import fixture from '../fixtures/result.sample.json';

// The API v2 draft `done` response copied byte-for-byte from
// backend/src/fixtures/result.sample.json (case 10 recorded ML predictions:
// 29 teeth, two lesions — FDI 45 and 26, both distal). Its image_base64 is a
// 1x1 placeholder, so it is only good for data/view-model tests.
export const v2DoneResponse: unknown = fixture;

export function loadV2Data(): InferenceData {
  const parsed = parseProcessResponse(v2DoneResponse);
  if (parsed.status !== 'done') throw new Error('v2 fixture must parse as a done response');
  return parsed.data;
}
