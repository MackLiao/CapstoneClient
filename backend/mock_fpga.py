"""Mock FPGA server for development. Returns numpy matmul results with configurable delay."""

import asyncio
import os

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock FPGA Server")

DELAY = float(os.environ.get("MOCK_FPGA_DELAY", "0.5"))


class MultiplyRequest(BaseModel):
    matrix_a: list[list[float]]
    matrix_b: list[list[float]]


class MultiplyResponse(BaseModel):
    result: list[list[float]]
    conversion_error: float = 0.0


@app.post("/multiply", response_model=MultiplyResponse)
async def multiply(req: MultiplyRequest):
    A = np.array(req.matrix_a, dtype=np.float64)
    B = np.array(req.matrix_b, dtype=np.float64)

    await asyncio.sleep(DELAY)

    C = np.matmul(A, B)

    return MultiplyResponse(
        result=C.tolist(),
        conversion_error=0.0,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
