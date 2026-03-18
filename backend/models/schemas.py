from pydantic import BaseModel, Field
from enum import Enum


class MatrixGenerateRequest(BaseModel):
    rows: int = Field(gt=0, le=16384)
    cols: int = Field(gt=0, le=16384)
    seed: int | None = None


class MatrixInfo(BaseModel):
    id: str
    rows: int
    cols: int


class MultiplyRequest(BaseModel):
    matrix_a_id: str
    matrix_b_id: str
    tile_size: int = 256


class JobInfo(BaseModel):
    job_id: str


class TileStatus(str, Enum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    FAILED = "failed"


class TileStatusEvent(BaseModel):
    i: int
    j: int
    k: int
    status: TileStatus
    completed: int
    total: int
    elapsed: float


class JobCompleteEvent(BaseModel):
    status: str  # "completed" or "failed"
    completed: int
    total: int
    elapsed: float


class VerificationResult(BaseModel):
    max_abs_error: float
    mean_abs_error: float
    passed: bool
    tolerance: float
