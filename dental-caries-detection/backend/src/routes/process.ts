import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { promises as fs } from 'node:fs';
import {
  saveInputImage,
  readResultJson,
  inputPath,
  cleanupJobFiles,
  resolveResultPath,
} from '../lib/imageStore.js';
import { dispatchInference } from '../services/mlClient.js';
import { adaptInferenceResult } from '../services/resultAdapter.js';
import { failJob, getLatestJob, startJob } from '../db/jobs.js';

// Wire format is frozen by docs-md/api-contract-v1.md (v1.1): POST errors are
// always 422 + status/fail_message, GET always answers one of
// idle/processing/fail/done, and `data` is the validated API v1 adapter output.

export async function registerProcessRoutes(app: FastifyInstance) {
  app.post('/process', async (req: FastifyRequest, reply: FastifyReply) => {
    let jobId: number | null = null;
    try {
      const data = await req.file();
      if (!data) {
        return reply.status(422).send({ status: 'fail', fail_message: 'No image provided.' });
      }

      // Supersede whatever was running and record the new job before telling
      // ml-service, so its WHERE status='processing' guard already sees it.
      const started = await startJob();
      jobId = started.jobId;
      const { replacedJobIds } = started;

      await saveInputImage(jobId, data.file);

      const accepted = await dispatchInference(jobId);
      if (!accepted) {
        await failJob(jobId, 'ML service rejected the job.');
        return reply
          .status(422)
          .send({ status: 'fail', fail_message: 'ML service rejected the job.' });
      }

      // Files of replaced jobs are no longer reachable through GET /process.
      await Promise.all(replacedJobIds.map(cleanupJobFiles));

      return reply.status(202).send({ status: 'processing' });
    } catch (err) {
      req.log.error(err);
      if (jobId !== null) {
        // If the row was already inserted, don't leave it 'processing' forever.
        await failJob(jobId, 'Failed to upload image.').catch(() => {});
        await cleanupJobFiles(jobId);
      }
      return reply.status(422).send({ status: 'fail', fail_message: 'Failed to upload image.' });
    }
  });

  app.get('/process', async (req: FastifyRequest, reply: FastifyReply) => {
    reply.header('Cache-Control', 'no-store');

    let job;
    try {
      job = await getLatestJob();
    } catch (err) {
      // The frontend retries failed polls with backoff, so a transient DB
      // outage shows as "reconnecting" instead of a terminal failure.
      req.log.error(err, 'GET /process: status lookup failed');
      return reply.status(503).send({ fail_message: 'Status database unavailable.' });
    }

    if (!job) {
      return reply.send({ status: 'idle' });
    }

    switch (job.status) {
      case 'processing':
      case 'superseded':
        return reply.send({ status: 'processing' });

      case 'fail':
        return reply.send({ status: 'fail', fail_message: job.failMessage || 'Inference failed.' });

      case 'done':
        let resultJson: unknown;
        try {
          resultJson = await readResultJson(resolveResultPath(job.id, job.resultPath));
        } catch (err) {
          req.log.error(err, `GET /process: result for job ${job.id} unreadable`);
          return reply.send({ status: 'fail', fail_message: 'Result file unavailable.' });
        }

        let data;
        try {
          data = adaptInferenceResult(resultJson);
        } catch (err) {
          req.log.error(err, `GET /process: invalid result for job ${job.id}`);
          return reply.send({ status: 'fail', fail_message: 'Invalid inference result.' });
        }

        try {
          const imgBuf = await fs.readFile(inputPath(job.id));
          return reply.send({
            status: 'done',
            image_base64: `data:image/jpeg;base64,${imgBuf.toString('base64')}`,
            data,
          });
        } catch (err) {
          req.log.error(err, `GET /process: input image for job ${job.id} unreadable`);
          return reply.send({ status: 'fail', fail_message: 'Result file unavailable.' });
        }
    }
  });
}
