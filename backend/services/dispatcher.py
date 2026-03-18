import asyncio
import logging

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
    Payload: {"matrix_a": [[...]], "matrix_b": [[...]]}
    Response: {"result": [[...]]}

    Retries on timeout with exponential backoff.
    """
    payload = {
        "matrix_a": a_tile.tolist(),
        "matrix_b": b_tile.tolist(),
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
            return np.array(data["result"], dtype=np.float64)
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
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
