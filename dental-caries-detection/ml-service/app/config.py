from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """ML Service configuration settings."""
    ML_SERVICE_PORT: int = 8000
    ML_SERVICE_HOST: str = "0.0.0.0"
    MODEL_PATH: str = str(Path(__file__).parent.parent / "models" / "best.pt")
    CONFIDENCE_THRESHOLD: float = 0.25
    MODEL_VERSION: str = "v1.0.0"
    LOG_LEVEL: str = "info"
    MAX_FILE_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: set = {"image/jpeg", "image/png"}

    class Config:
        env_file = ".env"


settings = Settings()
