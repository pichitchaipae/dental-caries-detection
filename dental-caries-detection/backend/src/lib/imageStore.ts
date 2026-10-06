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
import { join, resolve, sep } from 'node:path';
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
export async function saveInputImage(jobId: number, stream: Readable): Promise<string> {
  const dest = inputPath(jobId);
  await pipeline(stream, createWriteStream(dest));
  return dest;
}

/**
 * Where to read a finished job's result: the `result_path` column ml-service
 * wrote, as long as it points inside SHARED_DIR; otherwise the conventional name.
 */
export function resolveResultPath(jobId: number, storedResultPath: string | null): string {
  if (storedResultPath) {
    const resolved = resolve(storedResultPath);
    if (resolved.startsWith(resolve(SHARED_DIR) + sep)) return resolved;
  }
  return resultPath(jobId);
}

/**
 * Read the inference result JSON for a completed job.
 * Pass the path from resolveResultPath().
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
