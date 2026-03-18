import logging
import time

import httpx
import numpy as np

from ..config import settings
from ..models.schemas import JobCompleteEvent, TileStatus, TileStatusEvent
from ..models.state import Job
from .accumulator import accumulate
from .dispatcher import dispatch_tile
from .slicer import pad_matrix, slice_matrix, strip_padding

logger = logging.getLogger(__name__)


async def run_job(job: Job) -> None:
    """Run the blocked matrix multiplication job sequentially."""

    T = job.tile_size
    M, K = job.matrix_a.shape
    _, N = job.matrix_b.shape

    # Pad matrices
    A_padded = pad_matrix(job.matrix_a, T)
    B_padded = pad_matrix(job.matrix_b, T)

    # Slice into tile grids
    A_tiles = slice_matrix(A_padded, T)
    B_tiles = slice_matrix(B_padded, T)

    # Init result matrix (padded size)
    C = np.zeros((A_padded.shape[0], B_padded.shape[1]), dtype=np.float64)

    async with httpx.AsyncClient() as client:
        for i in range(job.I):
            for j in range(job.J):
                for k in range(job.K_tiles):
                    if job.cancelled:
                        logger.info(f"Job {job.job_id} cancelled")
                        job.finished = True
                        await job.event_queue.put(None)
                        return

                    # Mark in-flight
                    job.set_tile_status(i, j, k, TileStatus.IN_FLIGHT)
                    elapsed = time.time() - job.start_time
                    await job.event_queue.put(
                        TileStatusEvent(
                            i=i, j=j, k=k,
                            status=TileStatus.IN_FLIGHT,
                            completed=job.completed_count,
                            total=job.total_ops,
                            elapsed=elapsed,
                        )
                    )

                    try:
                        result_tile = await dispatch_tile(
                            client,
                            A_tiles[i][k],
                            B_tiles[k][j],
                            settings.FPGA_URL,
                            timeout=settings.REQUEST_TIMEOUT,
                            max_retries=settings.MAX_RETRIES,
                            backoff_base=settings.RETRY_BACKOFF_BASE,
                        )
                        accumulate(C, result_tile, i, j, T)
                        job.set_tile_status(i, j, k, TileStatus.COMPLETED)
                        status = TileStatus.COMPLETED
                    except Exception as e:
                        logger.error(f"Tile ({i},{j},{k}) failed: {e}")
                        job.set_tile_status(i, j, k, TileStatus.FAILED)
                        status = TileStatus.FAILED
                        if settings.STOP_ON_FAILURE:
                            job.finished = True
                            await job.event_queue.put(
                                TileStatusEvent(
                                    i=i, j=j, k=k,
                                    status=status,
                                    completed=job.completed_count,
                                    total=job.total_ops,
                                    elapsed=time.time() - job.start_time,
                                )
                            )
                            await job.event_queue.put(None)
                            return

                    elapsed = time.time() - job.start_time
                    await job.event_queue.put(
                        TileStatusEvent(
                            i=i, j=j, k=k,
                            status=status,
                            completed=job.completed_count,
                            total=job.total_ops,
                            elapsed=elapsed,
                        )
                    )

    # Strip padding and store result
    job.result = strip_padding(C, M, N)
    job.finished = True
    elapsed = time.time() - job.start_time
    await job.event_queue.put(
        JobCompleteEvent(
            status="completed",
            completed=job.completed_count,
            total=job.total_ops,
            elapsed=elapsed,
        )
    )
    await job.event_queue.put(None)
    logger.info(f"Job {job.job_id} completed in {elapsed:.1f}s")
