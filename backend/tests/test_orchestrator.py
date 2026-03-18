import asyncio

import numpy as np
import pytest

from backend.models.schemas import TileStatus, TileStatusEvent, JobCompleteEvent
from backend.models.state import Job
from backend.services.orchestrator import run_job


class MockFPGAClient:
    """Mock httpx.AsyncClient that returns numpy matmul results."""

    def __init__(self, delay: float = 0.0, fail_tiles: set | None = None):
        self.delay = delay
        self.fail_tiles = fail_tiles or set()
        self.call_count = 0

    async def post(self, url, json=None, timeout=None):
        self.call_count += 1
        A = np.array(json["matrix_a"])
        B = np.array(json["matrix_b"])

        if self.call_count in self.fail_tiles:
            raise Exception("Mock FPGA failure")

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        return MockResponse(np.matmul(A, B).tolist())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockResponse:
    def __init__(self, result):
        self._result = result
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"result": self._result}


@pytest.mark.asyncio
async def test_full_pipeline_small(monkeypatch):
    """8x8 matrix, tile_size=4 → should match np.matmul exactly."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((8, 8))
    B = rng.standard_normal((8, 8))

    job = Job(A, B, tile_size=4)

    # Monkeypatch dispatcher to use mock client
    mock_client = MockFPGAClient()

    import backend.services.orchestrator as orch_module
    original_dispatch = orch_module.dispatch_tile if hasattr(orch_module, 'dispatch_tile') else None

    async def mock_dispatch(client, a_tile, b_tile, fpga_url, timeout=60, max_retries=3, backoff_base=2.0):
        response = await mock_client.post("", json={"matrix_a": a_tile.tolist(), "matrix_b": b_tile.tolist()})
        return np.array(response.json()["result"])

    monkeypatch.setattr("backend.services.orchestrator.dispatch_tile", mock_dispatch)

    await run_job(job)

    assert job.finished
    assert job.result is not None

    expected = np.matmul(A, B)
    np.testing.assert_allclose(job.result, expected, atol=1e-10)

    # 2x2x2 = 8 tile operations
    assert job.completed_count == 8


@pytest.mark.asyncio
async def test_non_square_matrix(monkeypatch):
    """Test rectangular matrices: (6x10) * (10x4), tile_size=4."""
    rng = np.random.default_rng(123)
    A = rng.standard_normal((6, 10))
    B = rng.standard_normal((10, 4))

    job = Job(A, B, tile_size=4)

    mock_client = MockFPGAClient()

    async def mock_dispatch(client, a_tile, b_tile, fpga_url, timeout=60, max_retries=3, backoff_base=2.0):
        response = await mock_client.post("", json={"matrix_a": a_tile.tolist(), "matrix_b": b_tile.tolist()})
        return np.array(response.json()["result"])

    monkeypatch.setattr("backend.services.orchestrator.dispatch_tile", mock_dispatch)

    await run_job(job)

    assert job.finished
    expected = np.matmul(A, B)
    np.testing.assert_allclose(job.result, expected, atol=1e-10)
    assert job.result.shape == (6, 4)


@pytest.mark.asyncio
async def test_cancellation(monkeypatch):
    """Cancel job mid-execution."""
    A = np.ones((8, 8))
    B = np.ones((8, 8))
    job = Job(A, B, tile_size=4)

    call_count = 0

    async def mock_dispatch(client, a_tile, b_tile, fpga_url, timeout=60, max_retries=3, backoff_base=2.0):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            job.cancelled = True
        return np.matmul(a_tile, b_tile)

    monkeypatch.setattr("backend.services.orchestrator.dispatch_tile", mock_dispatch)

    await run_job(job)

    assert job.finished
    assert job.cancelled
    # Should have stopped before completing all 8 tiles
    assert job.completed_count < 8
