import pytest
import uuid
from schemas.inference import PipelineError, InferenceResponse

def test_pipeline_error_serialization():
    err = PipelineError(
        code="RF_FEATURE_SCHEMA_MISMATCH",
        message="Missing features",
        detection_id="det_0",
        fdi="46"
    )
    
    assert err.code == "RF_FEATURE_SCHEMA_MISMATCH"
    assert err.detection_id == "det_0"
    
    dump = err.model_dump()
    assert dump["code"] == "RF_FEATURE_SCHEMA_MISMATCH"
