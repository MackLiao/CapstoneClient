import asyncio
import logging
import uuid

import httpx
import numpy as np

logger = logging.getLogger(__name__)


async def dispatch_tile(
    client: httpx.AsyncClient,
    a_tile: np.ndarray,
    b_tile: np.ndarray,
    fpga_url: str,
    timeout: int = 60,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> np.ndarray:
    """Send two tiles to the FPGA for multiplication and return the result.

    Template endpoint: POST {fpga_url}/multiply
    Payload: {"A": [[...]], "B": [[...]], "request_id": "...", "return_mode": "full"}
    Response: {"request_id": ..., "status": "ok", "result": {"C": [[...]]}, ...}

    Retries on timeout with exponential backoff.
    """
    request_id = str(uuid.uuid4())
    payload = {
        "A": a_tile.tolist(),
        "B": b_tile.tolist(),
        "request_id": request_id,
        "return_mode": "full",
    }

    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.post(
                f"{fpga_url}/multiply",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "error":
                raise RuntimeError(
                    f"FPGA returned error: {data.get('error', 'unknown')}"
                )

            return np.array(data["result"]["C"], dtype=np.float64)
        except (httpx.TimeoutException, httpx.HTTPStatusError, RuntimeError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait = backoff_base ** (attempt + 1)
                logger.warning(
                    f"FPGA dispatch attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    f"FPGA dispatch failed after {max_retries} attempts: {e}"
                )

    raise last_exception  # type: ignore[misc]
