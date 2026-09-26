import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { saveInputImage, readResultJson, resultPath, inputPath, cleanupJobFiles } from '../lib/imageStore.js';
import { dispatchInference } from '../services/mlClient.js';
import { promises as fs } from 'node:fs';
import pg from 'pg';

let currentJobId: number | null = null;
let currentJobStatus: 'idle' | 'processing' | 'done' | 'fail' = 'idle';

export async function registerProcessRoutes(app: FastifyInstance) {
  app.post('/process', async (req: FastifyRequest, reply: FastifyReply) => {
    try {
      const data = await req.file();
      if (!data) {
        return reply.status(422).send({ fail_message: 'No image provided.' });
      }

      // Generate a simple job ID based on timestamp
      const jobId = Date.now();
      await saveInputImage(jobId, data.file);

      // Insert job into database
      const client = new pg.Client({ connectionString: process.env.DATABASE_URL });
      try {
        await client.connect();
        await client.query(
          "INSERT INTO jobs (id, status, updated_at) VALUES ($1, 'processing', now())",
          [jobId]
        );
      } catch (dbErr) {
        req.log.error('DB insert error: ' + dbErr);
      } finally {
        await client.end();
      }

      const accepted = await dispatchInference(jobId);
      if (!accepted) {
        return reply.status(422).send({ fail_message: 'ML service rejected the job.' });
      }

      currentJobId = jobId;
      currentJobStatus = 'processing';
      return reply.status(202).send({ status: 'processing' });
    } catch (err) {
      req.log.error(err);
      return reply.status(422).send({ fail_message: 'Failed to upload image.' });
    }
  });

  app.get('/process', async (req: FastifyRequest, reply: FastifyReply) => {
    if (currentJobStatus === 'idle' || !currentJobId) {
      return reply.send({ status: 'idle' });
    }

    if (currentJobStatus === 'processing') {
      try {
        // Try to read the result JSON to see if ml-service finished
        const resultFile = resultPath(currentJobId);
        const stat = await fs.stat(resultFile).catch(() => null);
        
          if (stat) {
          const resultJson = await readResultJson(resultFile);
          
          // Read input image as base64
          const imgFile = inputPath(currentJobId);
          const imgBuf = await fs.readFile(imgFile);
          const base64 = `data:image/jpeg;base64,${imgBuf.toString('base64')}`;

          currentJobStatus = 'done';
          return reply.send({
            status: 'done',
            image_base64: base64,
            data: resultJson
          });
        } else {
          // Check DB if it was failed by ml-service
          const client = new pg.Client({ connectionString: process.env.DATABASE_URL });
          try {
            await client.connect();
            const res = await client.query('SELECT status, fail_message FROM jobs WHERE id = $1', [currentJobId]);
            if (res.rows.length > 0) {
              const rowStatus = res.rows[0].status;
              if (rowStatus === 'fail') {
                currentJobStatus = 'fail';
                return reply.send({ status: 'fail', fail_message: res.rows[0].fail_message || 'Inference failed.' });
              }
            }
          } catch (e) {
            req.log.error('DB poll error: ' + e);
          } finally {
            await client.end();
          }

          return reply.send({ status: 'processing' });
        }
      } catch (err) {
        req.log.error(err);
        currentJobStatus = 'fail';
        return reply.send({ status: 'fail', fail_message: 'Failed to read inference results.' });
      }
    }
    
    // If it was already done (shouldn't happen on polling if frontend stops, but just in case)
    if (currentJobStatus === 'done' && currentJobId) {
      const resultFile = resultPath(currentJobId);
      const resultJson = await readResultJson(resultFile);
      const imgFile = inputPath(currentJobId);
      const imgBuf = await fs.readFile(imgFile);
      const base64 = `data:image/jpeg;base64,${imgBuf.toString('base64')}`;
      return reply.send({
        status: 'done',
        image_base64: base64,
        data: resultJson
      });
    }

    return reply.send({ status: currentJobStatus });
  });
}
