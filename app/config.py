import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "BBCA Chronos Intelligence"
    HF_MODEL_ID: str = "SkibidiBreaddd/BBCA-Chronos-14062026-v0"
    TICKER: str = "BBCA.JK"
    CONTEXT_WINDOW: int = 512
    HORIZON: int = 5
    BATCH_SIZE: int = 32
    HF_HOME: str = os.getenv("HF_HOME", "/root/.cache/huggingface")

    class Config:
        env_file = ".env"

settings = Settings()
