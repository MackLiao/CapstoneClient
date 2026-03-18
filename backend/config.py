from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    FPGA_URL: str = "http://localhost:5001"
    TILE_SIZE: int = 256
    REQUEST_TIMEOUT: int = 60
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE: float = 2.0
    MAX_MATRIX_DIM: int = 16384
    STOP_ON_FAILURE: bool = False

    model_config = {"env_prefix": "CAPSTONE_"}


settings = Settings()
