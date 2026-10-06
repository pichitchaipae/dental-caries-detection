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
import { failJob, getLatestJob, startJob } from '../db/jobs.js';

// Wire format is frozen by docs-md/api-contract-v1.md (v1.1): POST errors are
// always 422 + fail_message, GET always answers one of idle/processing/fail/done,
// and `data` is the ml-service result forwarded untouched.

export async function registerProcessRoutes(app: FastifyInstance) {
  app.post('/process', async (req: FastifyRequest, reply: FastifyReply) => {
    let jobId: number | null = null;
    try {
      const data = await req.file();
      if (!data) {
        return reply.status(422).send({ fail_message: 'No image provided.' });
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
        return reply.status(422).send({ fail_message: 'ML service rejected the job.' });
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
      return reply.status(422).send({ fail_message: 'Failed to upload image.' });
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
        try {
          const resultJson = await readResultJson(resolveResultPath(job.id, job.resultPath));
          const imgBuf = await fs.readFile(inputPath(job.id));
          return reply.send({
            status: 'done',
            image_base64: `data:image/jpeg;base64,${imgBuf.toString('base64')}`,
            data: resultJson,
          });
        } catch (err) {
          req.log.error(err, `GET /process: result for job ${job.id} unreadable`);
          return reply.send({ status: 'fail', fail_message: 'Result file unavailable.' });
        }
    }
  });
}
