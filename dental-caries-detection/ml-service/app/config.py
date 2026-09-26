from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """ML Service configuration settings."""
    database_url: str = Field(alias="DATABASE_URL")
    ml_service_port: int = Field(8001, alias="ML_SERVICE_PORT")
    ml_service_host: str = Field("0.0.0.0", alias="ML_SERVICE_HOST")
    shared_dir: str = Field("/shared", alias="SHARED_DIR")
    weights_dir: str = Field("/weights", alias="WEIGHTS_DIR")
    device: str = Field("cpu", alias="ML_DEVICE")
    caries_conf: float = Field(0.005, alias="CARIES_CONF")
    detection_threshold: float = Field(0.25, alias="DETECTION_THRESHOLD")
    enable_crop_segmenter: bool = Field(True, alias="ENABLE_CROP_SEGMENTER")
    log_level: str = Field("info", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        populate_by_name = True

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
