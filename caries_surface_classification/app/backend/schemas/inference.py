from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class PipelineError(BaseModel):
    code: str
    message: str
    detection_id: Optional[str] = None
    fdi: Optional[str] = None
    required_artifact: Optional[str] = None

class Point(BaseModel):
    x: float
    y: float

class ToothDetectionResult(BaseModel):
    detection_id: str
    fdi: str
    bbox: BoundingBox
    confidence: float
    original_class: str
    tooth_pts: List[List[float]] = Field(default_factory=list) # Polygons of tooth
    caries_pts: List[List[float]] = Field(default_factory=list) # Polygons of caries (if found)
    is_valid: bool = True

class FeatureExtractionResult(BaseModel):
    detection_id: str
    fdi: str
    feature_names: List[str]
    feature_vector: List[float]
    feature_count: int
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)

class RFClassificationResult(BaseModel):
    predicted_surface: str
    rf_confidence: float
    method: str
    fallback_used: bool
    fallback_reason: Optional[str] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)

class FinalFinding(BaseModel):
    detection_id: str
    fdi: str
    surface: str
    bbox_xyxy: List[int]
    yolo_confidence: float
    rf_confidence: float
    class_probabilities: Dict[str, float]

class PipelineSummary(BaseModel):
    raw_detection_count: int = 0
    valid_detection_count: int = 0
    invalid_detection_count: int = 0
    duplicate_count: int = 0
    inference_latency_ms: int = 0

class InferenceResponse(BaseModel):
    request_id: str
    status: str
    pipeline_version: str = "reference-pipeline-version"
    summary: PipelineSummary
    findings: List[FinalFinding] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[PipelineError] = Field(default_factory=list)
    # All detected tooth bboxes [x1,y1,x2,y2,fdi] — used by renderer for grey boxes
    all_tooth_bboxes: Optional[List[List]] = Field(default=None, exclude=True)
