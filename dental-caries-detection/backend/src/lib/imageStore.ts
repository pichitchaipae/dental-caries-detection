/**
 * Image store — saves uploaded OPG images to the shared volume
 * using job-specific filenames to prevent race conditions.
 *
 * File naming convention:
 *   input-{jobId}.jpg   ← written here
 *   result-{jobId}.json ← written by ml-service; path stored in DB
 *
 * The shared directory is mounted at SHARED_DIR (default: /shared).
 */

import { createWriteStream, promises as fs } from 'node:fs';
import { join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import type { Readable } from 'node:stream';

const SHARED_DIR = process.env.SHARED_DIR ?? '/shared';

export function inputPath(jobId: number): string {
  return join(SHARED_DIR, `input-${jobId}.jpg`);
}

export function resultPath(jobId: number): string {
  return join(SHARED_DIR, `result-${jobId}.json`);
}

/**
 * Write an image stream to /shared/input-{jobId}.jpg.
 * Overwrites if a previous file exists (should not happen with single-concurrency).
 */
export async function saveInputImage(
  jobId: number,
  stream: Readable,
): Promise<string> {
  const dest = inputPath(jobId);
  await pipeline(stream, createWriteStream(dest));
  return dest;
}

/**
 * Read the inference result JSON for a completed job.
 * Expects the path stored in the `result_path` DB column — does NOT hardcode.
 */
export async function readResultJson(storedResultPath: string): Promise<unknown> {
  const raw = await fs.readFile(storedResultPath, 'utf-8');
  return JSON.parse(raw);
}

/**
 * Remove job-specific files from the shared volume.
 * Called when a job is superseded or fails, to avoid stale file accumulation.
 */
export async function cleanupJobFiles(jobId: number): Promise<void> {
  const files = [inputPath(jobId), resultPath(jobId)];
  await Promise.allSettled(files.map((f) => fs.unlink(f)));
}
