from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """ML Service configuration settings."""
    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True)

    database_url: str = Field(alias="DATABASE_URL")
    ml_service_port: int = Field(8001, alias="ML_SERVICE_PORT")
    ml_service_host: str = Field("0.0.0.0", alias="ML_SERVICE_HOST")
    shared_dir: str = Field("/shared", alias="SHARED_DIR")
    weights_dir: str = Field("/weights", alias="WEIGHTS_DIR")
    device: str = Field("cpu", alias="ML_DEVICE")
    # Crop-trained 1-class caries YOLO-seg.  0.25 is the starting operating
    # point; recalibrate on the new validation split before shipping.
    caries_conf: float = Field(0.25, alias="CARIES_CONF")
    caries_iou: float = Field(0.50, alias="CARIES_IOU")
    detection_threshold: float = Field(0.25, alias="DETECTION_THRESHOLD")
    # Crop-Seg is the required second stage in the target pipeline.  When it is
    # disabled the service falls back to pano masks (kept for local debugging).
    enable_crop_segmenter: bool = Field(True, alias="ENABLE_CROP_SEGMENTER")
    crop_pad: int = Field(20, alias="CROP_PAD")
    dedupe_iou: float = Field(0.50, alias="DEDUPE_IOU")
    # Black out other detected teeth inside the crop so a neighbour's lesion is
    # never learned/served as belonging to this tooth. Training uses the same
    # rule (scripts/prepare_caries_crop_dataset.py).
    suppress_neighbours: bool = Field(True, alias="SUPPRESS_NEIGHBOURS")
    # Deterministic surface heuristic zones, validated against the 500-case XML
    # annotations (see docs-md/pipeline-v2-caries-crop.md).
    surface_left_bound: float = Field(0.30, alias="SURFACE_LEFT_BOUND")
    surface_right_bound: float = Field(0.70, alias="SURFACE_RIGHT_BOUND")
    log_level: str = Field("info", alias="LOG_LEVEL")

    def input_path(self, job_id: int):
        from pathlib import Path
        return Path(self.shared_dir) / f"input-{job_id}.jpg"

    def result_tmp_path(self, job_id: int):
        from pathlib import Path
        return Path(self.shared_dir) / f"result-{job_id}.tmp"

    def result_path(self, job_id: int):
        from pathlib import Path
        return Path(self.shared_dir) / f"result-{job_id}.json"

settings = Settings()
