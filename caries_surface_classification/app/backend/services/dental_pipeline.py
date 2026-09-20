import time
import logging
from typing import List

from schemas.inference import (
    InferenceResponse, PipelineSummary, FinalFinding, PipelineError,
    ToothDetectionResult
)
from services.tooth_detection import ToothDetectionService
from services.caries_detection import CariesDetectionService
from services.feature_extraction import FeatureExtractor
from services.rf_classifier import RFClassifierService

logger = logging.getLogger(__name__)


class DentalPipelineOrchestrator:
    """
    Full dental pipeline:
      1. Tooth detection (all teeth + segmentation masks, via Tooth_seg_pano model)
      2. Caries detection (only teeth with lesions, via caries_detect.pt YOLO model)
      3. Feature extraction + RF surface classification (Occlusal / Mesial / Distal)
         only for teeth that actually have caries detections.

    Teeth without caries detections are included in the raw count and
    rendered with bounding boxes in the annotated image but do NOT appear
    in 'findings' (i.e. they are NOT sent to the RF classifier).
    """

    def __init__(self):
        self.tooth_detector   = ToothDetectionService()
        self.caries_detector  = CariesDetectionService()
        self.rf_classifier    = RFClassifierService()

    # ------------------------------------------------------------------
    def process_dental_radiograph(
        self, image_path: str, request_id: str
    ) -> InferenceResponse:

        start_time = time.perf_counter()
        findings: List[FinalFinding]  = []
        errors:   List[PipelineError] = []

        logger.info({
            "request_id": request_id,
            "stage": "pipeline_started",
            "status": "success",
            "message": "Processing started for dental radiograph",
        })

        try:
            # ── 1. Tooth Detection ────────────────────────────────────
            detections = self.tooth_detector.detect_teeth(image_path)
            raw_detection_count = len(detections)

            logger.info({
                "request_id": request_id,
                "stage": "tooth_detection",
                "status": "success",
                "detection_count": raw_detection_count,
            })

            if not detections:
                end_time = time.perf_counter()
                return InferenceResponse(
                    request_id=request_id,
                    status="success",
                    summary=PipelineSummary(
                        raw_detection_count=0,
                        valid_detection_count=0,
                        invalid_detection_count=0,
                        duplicate_count=0,
                        inference_latency_ms=int((end_time - start_time) * 1000),
                    ),
                    findings=[],
                    warnings=["No teeth detected in the image."],
                    errors=[],
                )

            # ── 2. Caries Detection ───────────────────────────────────
            # Returns dict: detection_id → List[[x,y]] caries point cloud
            # Only teeth that actually have caries lesions are keyed here.
            # Surface classification (Occlusal/Mesial/Distal) is NOT done
            # here — that is the RF model's job in step 3.
            try:
                caries_map = self.caries_detector.detect_caries(image_path, detections)
                logger.info({
                    "request_id": request_id,
                    "stage": "caries_detection",
                    "status": "success",
                    "teeth_with_caries": len(caries_map),
                })
            except FileNotFoundError as e:
                # caries_detect.pt missing — fallback: use tooth_pts as caries proxy
                logger.warning({
                    "request_id": request_id,
                    "stage": "caries_detection",
                    "status": "fallback",
                    "message": str(e),
                })
                errors.append(PipelineError(
                    code="CARIES_MODEL_MISSING",
                    message=(
                        f"Caries detection model unavailable: {e}. "
                        "Falling back — all detected teeth will be classified."
                    ),
                    required_artifact="caries_detect.pt",
                ))
                # Fallback: use each tooth's segmentation mask as the caries point cloud
                caries_map: dict = {}
                for det in detections:
                    if det.tooth_pts:
                        proxy_pts = [[float(p[0]), float(p[1])] for p in det.tooth_pts]
                    else:
                        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
                        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        proxy_pts = [
                            [float(x1), float(y1)], [float(mx), float(y1)],
                            [float(x2), float(y1)], [float(x2), float(my)],
                            [float(x2), float(y2)], [float(mx), float(y2)],
                            [float(x1), float(y2)], [float(x1), float(my)],
                            [float(mx), float(my)],
                        ]
                    caries_map[det.detection_id] = proxy_pts

            # ── 3. Feature Extraction + RF Classification ─────────────
            # Only teeth whose detection_id appears in caries_map are classified.
            # caries_map values are already caries point clouds (List[[x,y]]).
            # Surface classification: Occlusal / Mesial / Distal via rf_classify_ml.pkl
            valid_detections   = 0
            invalid_detections = 0

            # Build a lookup from detection_id → ToothDetectionResult
            det_by_id = {d.detection_id: d for d in detections}

            for det_id, caries_pts in caries_map.items():
                det = det_by_id.get(det_id)
                if det is None:
                    continue

                if not det.tooth_pts:
                    invalid_detections += 1
                    continue

                if not caries_pts:
                    invalid_detections += 1
                    continue

                # Feature extraction (14 geometric features from reference pipeline)
                features = FeatureExtractor.extract_features(
                    det.fdi, det.tooth_pts, caries_pts
                )

                # RF surface classification: Occlusal / Mesial / Distal
                # Smart Fallback chain: RF → X-Thirds → Default
                rf_result = self.rf_classifier.classify_surface(
                    features=features,
                    tooth_id=det.fdi,
                    tooth_pts=det.tooth_pts,
                    caries_pts=caries_pts,
                )

                findings.append(FinalFinding(
                    detection_id=det.detection_id,
                    fdi=det.fdi,
                    surface=rf_result.predicted_surface,
                    bbox_xyxy=[det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                    yolo_confidence=round(det.confidence, 4),
                    rf_confidence=round(rf_result.rf_confidence, 4),
                    class_probabilities=rf_result.probabilities,
                ))
                valid_detections += 1

                logger.info({
                    "request_id": request_id,
                    "stage": "rf_classification",
                    "status": "success",
                    "detection_id": det.detection_id,
                    "fdi": det.fdi,
                    "predicted_surface": rf_result.predicted_surface,
                    "method": rf_result.method,
                    "fallback_used": rf_result.fallback_used,
                })

            end_time = time.perf_counter()
            inference_latency_ms = int((end_time - start_time) * 1000)

            # Build list of all tooth bboxes [x1,y1,x2,y2,fdi] for renderer (grey boxes)
            all_tooth_bboxes = [
                [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.fdi]
                for d in detections
            ]

            return InferenceResponse(
                request_id=request_id,
                status="success",
                summary=PipelineSummary(
                    raw_detection_count=raw_detection_count,
                    valid_detection_count=valid_detections,
                    invalid_detection_count=invalid_detections,
                    duplicate_count=0,
                    inference_latency_ms=inference_latency_ms,
                ),
                findings=findings,
                warnings=[],
                errors=errors,
                all_tooth_bboxes=all_tooth_bboxes,
            )

        except Exception as e:
            logger.error({
                "request_id": request_id,
                "stage": "pipeline_error",
                "status": "error",
                "message": str(e),
            })
            errors.append(PipelineError(code="INTERNAL_ERROR", message=str(e)))
            return self._build_error_response(request_id, start_time, errors)

    # ------------------------------------------------------------------
    def _build_error_response(
        self, request_id: str, start_time: float, errors: List[PipelineError]
    ) -> InferenceResponse:
        end_time = time.perf_counter()
        return InferenceResponse(
            request_id=request_id,
            status="error",
            summary=PipelineSummary(
                inference_latency_ms=int((end_time - start_time) * 1000)
            ),
            findings=[],
            warnings=[],
            errors=errors,
        )
