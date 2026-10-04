import { delay, http, HttpResponse } from 'msw';
import type { RawInferenceData } from '../domain/inference';
import { buildMockResult } from './resultFactory';

// Simulates the backend's real POST/GET /process behavior (backend/src/routes/
// process.ts): single job at a time (a new POST replaces the previous job), 202 on
// accept, a generic 422 for any rejected upload (no 413/415), a 10 MB upload limit,
// and a `done` payload carrying the raw ml-service result. Only this mock goes
// away when pointing the UI at the real backend.

const BASE_URL = import.meta.env.VITE_FRONTEND_API_BASE_URL ?? 'http://localhost:8000';
const MOCK_DELAY_MS = Number(import.meta.env.VITE_MOCK_ML_DELAY_MS ?? 4000);
const MOCK_FAILURE_RATE = Number(import.meta.env.VITE_MOCK_ML_FAILURE_RATE ?? 0);
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // backend @fastify/multipart fileSize limit

type JobState =
  | { status: 'idle' }
  | { status: 'processing'; token: symbol }
  | { status: 'done'; data: RawInferenceData; imageBase64: string }
  | { status: 'fail'; fail_message: string };

let job: JobState = { status: 'idle' };

export const handlers = [
  http.post(`${BASE_URL}/process`, async ({ request }) => {
    const formData = await request.formData();
    const image = formData.get('image');

    if (!(image instanceof File)) {
      return HttpResponse.json({ fail_message: 'No image provided.' }, { status: 422 });
    }
    if (image.size > MAX_UPLOAD_BYTES) {
      return HttpResponse.json({ fail_message: 'Failed to upload image.' }, { status: 422 });
    }

    // A fresh token per submission: preemption is "the last token wins", the
    // same guarantee project-structure.md Section 8.2 gives via `id = :job_id`.
    const token = Symbol('job');
    job = { status: 'processing', token };

    void (async () => {
      await delay(MOCK_DELAY_MS);
      if (job.status !== 'processing' || job.token !== token) return; // preempted

      if (Math.random() < MOCK_FAILURE_RATE) {
        job = { status: 'fail', fail_message: 'Mock ML: simulated failure during detection stage' };
        return;
      }

      const { data, imageBase64 } = await buildMockResult(image);
      if (job.status !== 'processing' || job.token !== token) return; // preempted mid-build
      job = { status: 'done', data, imageBase64 };
    })();

    return HttpResponse.json({ status: 'processing' }, { status: 202 });
  }),

  http.get(`${BASE_URL}/process`, () => {
    if (job.status === 'idle') return HttpResponse.json({ status: 'idle' });
    if (job.status === 'processing') return HttpResponse.json({ status: 'processing' });
    if (job.status === 'fail') {
      return HttpResponse.json({ status: 'fail', fail_message: job.fail_message });
    }
    return HttpResponse.json({ status: 'done', image_base64: job.imageBase64, data: job.data });
  }),
];
