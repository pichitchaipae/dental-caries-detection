"""
Pipeline orchestrator — sequences all stages and atomically publishes the result.

Atomic publish protocol:
  1. Write result to  /shared/result-{job_id}.tmp  (+ fsync)
  2. Pre-publish DB check  (optimization — primary guard is the SQL predicate)
  3. os.replace() → /shared/result-{job_id}.json   (atomic rename)
  4. UPDATE jobs SET status='done' WHERE id=:job_id AND status='processing'
  5. If DB update returned 0 rows → job was superseded → delete the .json

This function is synchronous and called exclusively from inside the child
inference process.  No asyncio needed — all work is CPU/GPU/IO bound.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.pipeline.detection import run_stage1
from app.pipeline.pca_alignment import run_stage2
from app.pipeline.surface_heuristic import findings_for_lesions
from app.pipeline.postprocess import build_tooth_result
from models.versions import MODEL_VERSIONS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(job_id: int, settings: Settings, models: dict) -> None:
    """
    Full inference pipeline.  Synchronous — CPU/GPU-bound.
    Raises on unrecoverable error (child exits non-zero; monitor handles DB fail update).
    """
    # Import DB here to avoid circular at module level
    from app import db as db_mod
    from app.db import get_engine, update_job_done, update_job_fail, get_job_status

    engine = get_engine(settings.database_url)

    input_path  = str(settings.input_path(job_id))
    tmp_path    = settings.result_tmp_path(job_id)
    result_path = settings.result_path(job_id)

    log.info("pipeline start: job_id=%d image=%s", job_id, input_path)

    try:
        # ----------------------------------------------------------------
        # Stage 1 — Detection
        # ----------------------------------------------------------------
        detections, image = run_stage1(
            image_path=input_path,
            models=models,
            detection_threshold=settings.detection_threshold,
            caries_conf=settings.caries_conf,
            caries_iou=settings.caries_iou,
            crop_pad=settings.crop_pad,
            dedupe_iou=settings.dedupe_iou,
            suppress_neighbours=settings.suppress_neighbours,
        )
        log.info("Stage 1 complete: %d teeth", len(detections))

        # ----------------------------------------------------------------
        # Stage 2 — PCA axes
        # ----------------------------------------------------------------
        axes_by_fdi = run_stage2(detections)
        log.info("Stage 2 complete")

        # ----------------------------------------------------------------
        # Stage 3 — deterministic surface heuristic
        #   A 1-class caries lesion is assigned to mesial/occlusal/distal
        #   from its centroid relative to the refined tooth mask.
        # ----------------------------------------------------------------
        findings_by_fdi: dict[int, list[dict]] = {}
        for det in detections:
            findings_by_fdi[det.fdi] = findings_for_lesions(
                fdi=det.fdi,
                tooth_polygon=det.tooth_polygon,
                lesions=det.caries_lesions,
                left_bound=settings.surface_left_bound,
                right_bound=settings.surface_right_bound,
            )
        log.info(
            "Stage 3 complete: %d teeth with surface findings",
            sum(1 for f in findings_by_fdi.values() if f),
        )
        surface_source = "geometry_heuristic"

        # ----------------------------------------------------------------
        # Build result JSON
        # ----------------------------------------------------------------
        img_h, img_w = image.shape[:2]
        teeth_results = []
        for tooth_id, det in enumerate(detections):
            axes = axes_by_fdi.get(det.fdi)
            axes_dict = (
                {
                    "major": list(axes.major),
                    "minor": list(axes.minor),
                    "rotation_deg": axes.rotation_deg,
                    "clamped": axes.clamped,
                }
                if axes is not None
                else None
            )
            surfaces = findings_by_fdi.get(det.fdi, [])
            tooth_entry = build_tooth_result(
                id=tooth_id,
                fdi=det.fdi,
                bbox_xywh=det.bbox_xywh,
                mask_polygon=det.tooth_polygon,
                pano_confidence=det.pano_confidence,
                axes=axes_dict,
                surface_findings=surfaces,
                caries_lesions=det.caries_lesions,
            )
            teeth_results.append(tooth_entry)

        caries_model = MODEL_VERSIONS.get("caries_detector")
        result: dict[str, Any] = {
            "meta": {
                "job_id": job_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "image_size": {"width": img_w, "height": img_h},
                "tooth_count": len(teeth_results),
                # Teeth with at least one caries surface finding.
                "caries_count": sum(1 for t in teeth_results if t["surfaces"]),
                "caries_lesion_count": sum(len(t["caries"]) for t in teeth_results),
                "models": {
                    "detector": MODEL_VERSIONS["pano_detector"]["filename"],
                    "classifier": (
                        caries_model["filename"] if caries_model else "not_configured"
                    ),
                },
            },
            "teeth": teeth_results,
        }

        # Provenance for teeth[].surfaces[]. `geometry_heuristic` is the only
        # serving path; the CNN/RF paths were removed from production.
        result["surface_source"] = surface_source

        # ----------------------------------------------------------------
        # Atomic publish
        # ----------------------------------------------------------------
        _atomic_publish(
            result=result,
            tmp_path=tmp_path,
            result_path=result_path,
            job_id=job_id,
            engine=engine,
            update_job_done=update_job_done,
            get_job_status=get_job_status,
        )

    except Exception as exc:
        # Clean up temp file if it exists
        tmp_path.unlink(missing_ok=True)
        # Short message to DB; full trace to application logs
        short_msg = repr(exc)[:255]
        log.exception("pipeline failed: job_id=%d", job_id)
        update_job_fail(engine, job_id, short_msg)
        raise  # child exits non-zero → monitor thread handles


def _atomic_publish(
    result: dict,
    tmp_path: Path,
    result_path: Path,
    job_id: int,
    engine: Any,
    update_job_done: Any,
    get_job_status: Any,
) -> None:
    """
    Write result atomically:
      tmp → fsync → pre-check → os.replace → DB update → cleanup on supersede
    """
    # 1. Write + fsync
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    log.debug("result written to tmp: %s", tmp_path)

    # 2. Pre-publish check (optimization — not the safety predicate)
    current_status = get_job_status(engine, job_id)
    if current_status is not None and current_status != "processing":
        tmp_path.unlink(missing_ok=True)
        log.warning(
            "job_id=%d status='%s' before publish — discarding result",
            job_id, current_status,
        )
        return

    # 3. Atomic rename (.tmp → .json)
    os.replace(tmp_path, result_path)
    log.info("result atomically renamed: %s", result_path)

    # 4. Update DB (primary safety guard: WHERE status='processing')
    updated = update_job_done(engine, job_id, str(result_path))
    if not updated and current_status is not None:
        # Superseded between rename and DB update
        result_path.unlink(missing_ok=True)
        log.warning(
            "job_id=%d DB update returned 0 rows (superseded) — result removed",
            job_id,
        )
    else:
        log.info("job_id=%d marked done in DB (or DB not present/row missing)", job_id)
