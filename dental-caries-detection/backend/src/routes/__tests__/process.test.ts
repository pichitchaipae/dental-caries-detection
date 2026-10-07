import { mkdtempSync, existsSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { adaptInferenceResult } from '../../services/resultAdapter.js';

// imageStore reads SHARED_DIR at import time; the app is imported dynamically below.
const sharedDir = mkdtempSync(join(tmpdir(), 'be-shared-'));
process.env.SHARED_DIR = sharedDir;

vi.mock('../../db/jobs.js', () => ({
  startJob: vi.fn(),
  getLatestJob: vi.fn(),
  failJob: vi.fn(),
}));
vi.mock('../../services/mlClient.js', () => ({
  dispatchInference: vi.fn(),
}));

const jobs = await import('../../db/jobs.js');
const ml = await import('../../services/mlClient.js');
const { buildApp } = await import('../../app.js');

const rawFixture = readFileSync(
  join(import.meta.dirname, '../../fixtures/ml-result.raw.sample.json'),
  'utf8'
);

function multipartBody(filename = 'opg.jpg', content = Buffer.from([0xff, 0xd8, 0xff, 0x00])) {
  const boundary = '----testboundary';
  const body = Buffer.concat([
    Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="image"; filename="${filename}"\r\nContent-Type: image/jpeg\r\n\r\n`
    ),
    content,
    Buffer.from(`\r\n--${boundary}--\r\n`),
  ]);
  return { body, headers: { 'content-type': `multipart/form-data; boundary=${boundary}` } };
}

describe('/process', () => {
  const app = buildApp();

  beforeEach(() => {
    vi.resetAllMocks();
    vi.mocked(jobs.failJob).mockResolvedValue(undefined);
  });

  afterAll(async () => {
    await app.close();
    rmSync(sharedDir, { recursive: true, force: true });
  });

  describe('GET', () => {
    it('is idle when no job exists, with no-store', async () => {
      vi.mocked(jobs.getLatestJob).mockResolvedValue(null);
      const res = await app.inject({ method: 'GET', url: '/process' });
      expect(res.statusCode).toBe(200);
      expect(res.headers['cache-control']).toBe('no-store');
      expect(res.json()).toEqual({ status: 'idle' });
    });

    it.each(['processing', 'superseded'] as const)('reports %s as processing', async (status) => {
      vi.mocked(jobs.getLatestJob).mockResolvedValue({
        id: 1,
        status,
        resultPath: null,
        failMessage: null,
      });
      const res = await app.inject({ method: 'GET', url: '/process' });
      expect(res.json()).toEqual({ status: 'processing' });
    });

    it('reports fail with the stored message', async () => {
      vi.mocked(jobs.getLatestJob).mockResolvedValue({
        id: 2,
        status: 'fail',
        resultPath: null,
        failMessage: "TypeError('boom')",
      });
      const res = await app.inject({ method: 'GET', url: '/process' });
      expect(res.json()).toEqual({ status: 'fail', fail_message: "TypeError('boom')" });
    });

    it('adapts the ml-service result to API v1 when done', async () => {
      const id = 3;
      writeFileSync(join(sharedDir, `input-${id}.jpg`), Buffer.from('img'));
      writeFileSync(join(sharedDir, `result-${id}.json`), rawFixture);
      vi.mocked(jobs.getLatestJob).mockResolvedValue({
        id,
        status: 'done',
        resultPath: join(sharedDir, `result-${id}.json`),
        failMessage: null,
      });
      const res = await app.inject({ method: 'GET', url: '/process' });
      const body = res.json();
      expect(body.status).toBe('done');
      expect(body.image_base64).toBe(
        `data:image/jpeg;base64,${Buffer.from('img').toString('base64')}`
      );
      expect(body.data).toEqual(adaptInferenceResult(JSON.parse(rawFixture)));
    });

    it('ignores a result_path outside SHARED_DIR', async () => {
      const id = 4;
      writeFileSync(join(sharedDir, `input-${id}.jpg`), Buffer.from('img'));
      writeFileSync(join(sharedDir, `result-${id}.json`), rawFixture);
      vi.mocked(jobs.getLatestJob).mockResolvedValue({
        id,
        status: 'done',
        resultPath: '/etc/passwd',
        failMessage: null,
      });
      const res = await app.inject({ method: 'GET', url: '/process' });
      expect(res.json().data).toEqual(adaptInferenceResult(JSON.parse(rawFixture)));
    });

    it('fails when a done job has no result file', async () => {
      vi.mocked(jobs.getLatestJob).mockResolvedValue({
        id: 5,
        status: 'done',
        resultPath: null,
        failMessage: null,
      });
      const res = await app.inject({ method: 'GET', url: '/process' });
      expect(res.json()).toEqual({ status: 'fail', fail_message: 'Result file unavailable.' });
    });

    it('answers 503 when the DB is unreachable, so the client retries', async () => {
      vi.mocked(jobs.getLatestJob).mockRejectedValue(new Error('ECONNREFUSED'));
      const res = await app.inject({ method: 'GET', url: '/process' });
      expect(res.statusCode).toBe(503);
    });
  });

  describe('POST', () => {
    it('starts a job, dispatches it and cleans up replaced jobs', async () => {
      const oldId = 10;
      writeFileSync(join(sharedDir, `input-${oldId}.jpg`), 'old');
      writeFileSync(join(sharedDir, `result-${oldId}.json`), '{}');
      vi.mocked(jobs.startJob).mockResolvedValue({ jobId: 11, replacedJobIds: [oldId] });
      vi.mocked(ml.dispatchInference).mockResolvedValue(true);

      const res = await app.inject({ method: 'POST', url: '/process', ...multipartBody() });

      expect(res.statusCode).toBe(202);
      expect(res.json()).toEqual({ status: 'processing' });
      const newId = 11;
      expect(vi.mocked(ml.dispatchInference)).toHaveBeenCalledWith(newId);
      expect(existsSync(join(sharedDir, `input-${newId}.jpg`))).toBe(true);
      expect(existsSync(join(sharedDir, `input-${oldId}.jpg`))).toBe(false);
      expect(existsSync(join(sharedDir, `result-${oldId}.json`))).toBe(false);
    });

    it('records the job before dispatching it to ml-service', async () => {
      const order: string[] = [];
      vi.mocked(jobs.startJob).mockImplementation(async () => {
        order.push('startJob');
        return { jobId: 13, replacedJobIds: [] };
      });
      vi.mocked(ml.dispatchInference).mockImplementation(async () => {
        order.push('dispatch');
        return true;
      });
      await app.inject({ method: 'POST', url: '/process', ...multipartBody() });
      expect(order).toEqual(['startJob', 'dispatch']);
    });

    it('marks the job failed when ml-service rejects it', async () => {
      vi.mocked(jobs.startJob).mockResolvedValue({ jobId: 12, replacedJobIds: [] });
      vi.mocked(ml.dispatchInference).mockResolvedValue(false);
      const res = await app.inject({ method: 'POST', url: '/process', ...multipartBody() });
      expect(res.statusCode).toBe(422);
      expect(res.json()).toEqual({
        status: 'fail',
        fail_message: 'ML service rejected the job.',
      });
      expect(vi.mocked(jobs.failJob)).toHaveBeenCalledWith(12, 'ML service rejected the job.');
    });

    it('answers 422 without dispatching when the DB write fails', async () => {
      vi.mocked(jobs.startJob).mockRejectedValue(new Error('db down'));
      const res = await app.inject({ method: 'POST', url: '/process', ...multipartBody() });
      expect(res.statusCode).toBe(422);
      expect(res.json()).toEqual({
        status: 'fail',
        fail_message: 'Failed to upload image.',
      });
      expect(vi.mocked(ml.dispatchInference)).not.toHaveBeenCalled();
    });

    it('marks the job failed and removes its file when dispatch throws', async () => {
      vi.mocked(jobs.startJob).mockResolvedValue({ jobId: 14, replacedJobIds: [] });
      vi.mocked(ml.dispatchInference).mockRejectedValue(new Error('boom'));
      const res = await app.inject({ method: 'POST', url: '/process', ...multipartBody() });
      expect(res.statusCode).toBe(422);
      expect(vi.mocked(jobs.failJob)).toHaveBeenCalledWith(14, 'Failed to upload image.');
      expect(existsSync(join(sharedDir, 'input-14.jpg'))).toBe(false);
    });

    it('answers 422 when no file is sent', async () => {
      const boundary = '----empty';
      const res = await app.inject({
        method: 'POST',
        url: '/process',
        headers: { 'content-type': `multipart/form-data; boundary=${boundary}` },
        body: `--${boundary}--\r\n`,
      });
      expect(res.statusCode).toBe(422);
      expect(vi.mocked(jobs.startJob)).not.toHaveBeenCalled();
    });
  });
});
