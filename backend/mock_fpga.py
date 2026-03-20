"""Mock FPGA server for development. Mimics the Flask app running on the PYNQ board."""

import asyncio
import os
import time
import uuid

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock FPGA Server")

DELAY = float(os.environ.get("MOCK_FPGA_DELAY", "0.5"))

# Simulated hardware tile dimensions
TILE_ROWS = 4
TILE_COLS = 4
TILE_DEPTH = 4


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tile_rows": TILE_ROWS,
        "tile_cols": TILE_COLS,
        "tile_depth": TILE_DEPTH,
    }


@app.post("/multiply")
async def multiply(request: Request):
    data = await request.json()

    request_id = data.get("request_id")

    def make_error(message: str, code: int = 400):
        return JSONResponse(
            status_code=code,
            content={
                "request_id": request_id,
                "status": "error",
                "mode": None,
                "shape": None,
                "elapsed_sec": None,
                "result": None,
                "stats": None,
                "verify": None,
                "error": message,
            },
        )

    if "A" not in data or "B" not in data:
        return make_error("Request must include 'A' and 'B' matrices")

    return_mode = data.get("return_mode", "auto")
    if return_mode not in ("auto", "full", "summary"):
        return make_error("return_mode must be one of: auto, full, summary")

    try:
        mat_a = np.array(data["A"], dtype=np.float32)
        mat_b = np.array(data["B"], dtype=np.float32)
    except (ValueError, TypeError) as e:
        return make_error(f"Invalid matrix data: {e}")

    if mat_a.ndim != 2 or mat_b.ndim != 2:
        return make_error("A and B must be 2D matrices")

    M, K_a = mat_a.shape
    K_b, N = mat_b.shape

    if K_a != K_b:
        return make_error(
            f"Inner dimensions must match: A is {list(mat_a.shape)}, B is {list(mat_b.shape)}"
        )

    await asyncio.sleep(DELAY)

    t0 = time.time()
    result = (mat_a @ mat_b).astype(np.float64)
    elapsed = round(time.time() - t0, 6)

    stats = {
        "sum": float(np.sum(result)),
        "max": float(np.max(result)),
        "min": float(np.min(result)),
    }

    return {
        "request_id": request_id,
        "status": "ok",
        "mode": "full",
        "shape": [int(M), int(N)],
        "elapsed_sec": elapsed,
        "result": {
            "C": result.tolist(),
        },
        "stats": stats,
        "verify": None,
        "error": None,
    }
