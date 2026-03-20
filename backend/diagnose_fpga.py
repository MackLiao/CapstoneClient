"""
Diagnostic script to test the FPGA board's multiply endpoint directly.
Run from the project root:
    python -m backend.diagnose_fpga [--url http://192.168.2.99:5000]
"""

import argparse
import sys

import httpx
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://192.168.2.99:5000")
    args = parser.parse_args()

    url = args.url
    print(f"Testing FPGA at {url}\n")

    # 1. Health check
    print("=" * 60)
    print("1. Health check")
    print("=" * 60)
    try:
        r = httpx.get(f"{url}/health", timeout=10)
        health = r.json()
        print(f"   Status: {health}")
        tile_rows = health.get("tile_rows")
        tile_cols = health.get("tile_cols")
        tile_depth = health.get("tile_depth")
        print(f"   Hardware dims: rows={tile_rows}, cols={tile_cols}, depth={tile_depth}")
    except Exception as e:
        print(f"   FAILED: {e}")
        sys.exit(1)

    # 2. Identity test: I @ I = I
    print()
    print("=" * 60)
    dim = tile_rows if tile_rows else 4
    print(f"2. Identity test ({dim}x{dim}): I @ I should = I")
    print("=" * 60)
    I_mat = np.eye(dim, dtype=np.float32)
    r = httpx.post(f"{url}/multiply", json={
        "A": I_mat.tolist(),
        "B": I_mat.tolist(),
        "return_mode": "full",
    }, timeout=30)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR from board: {data['error']}")
    else:
        C = np.array(data["result"]["C"])
        print(f"   Result shape: {C.shape}")
        print(f"   Result dtype values: min={C.min():.6f}, max={C.max():.6f}")
        print(f"   Expected: identity matrix (diag=1, off-diag=0)")
        print(f"   Max error vs identity: {np.max(np.abs(C - I_mat)):.6e}")
        print(f"   Result[0,:4] = {C[0,:4]}")

    # 3. Simple known multiplication
    print()
    print("=" * 60)
    print(f"3. Known multiplication: ones({dim},{dim}) @ ones({dim},{dim})")
    print(f"   Expected: every element = {dim}.0")
    print("=" * 60)
    A = np.ones((dim, dim), dtype=np.float32)
    B = np.ones((dim, dim), dtype=np.float32)
    r = httpx.post(f"{url}/multiply", json={
        "A": A.tolist(),
        "B": B.tolist(),
        "return_mode": "full",
    }, timeout=30)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR from board: {data['error']}")
    else:
        C = np.array(data["result"]["C"])
        expected = np.full((dim, dim), float(dim))
        print(f"   Result[0,0] = {C[0,0]:.6f}  (expected {dim}.0)")
        print(f"   Max error: {np.max(np.abs(C - expected)):.6e}")
        if abs(C[0, 0] - dim) > 1:
            ratio = C[0, 0] / dim
            print(f"   *** SCALING ISSUE: result/expected = {ratio:.2f}")
            print(f"   *** This suggests values are off by factor ~{ratio:.0f}")
            if ratio > 1:
                import math
                bits = math.log2(ratio)
                print(f"   *** That's ~2^{bits:.1f} — likely a fixed-point frac_bits issue")

    # 4. Random tile-sized test (matches what orchestrator sends)
    print()
    print("=" * 60)
    print("4. Random tile test (256x256) — same size orchestrator sends")
    print("=" * 60)
    rng = np.random.default_rng(42)
    A = rng.standard_normal((256, 256)).astype(np.float32)
    B = rng.standard_normal((256, 256)).astype(np.float32)
    golden = (A.astype(np.float64) @ B.astype(np.float64))

    r = httpx.post(f"{url}/multiply", json={
        "A": A.tolist(),
        "B": B.tolist(),
        "return_mode": "full",
    }, timeout=120)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR from board: {data['error']}")
        print(f"   *** The board may not support 256x256 tiles!")
        print(f"   *** Try reducing tile_size to match hardware dims")
    else:
        C = np.array(data["result"]["C"], dtype=np.float64)
        print(f"   Result shape: {C.shape}")
        diff = np.abs(C - golden)
        max_err = np.max(diff)
        mean_err = np.mean(diff)
        print(f"   Max abs error:  {max_err:.6e}")
        print(f"   Mean abs error: {mean_err:.6e}")
        print(f"   Golden[0,0]={golden[0,0]:.4f}  FPGA[0,0]={C[0,0]:.4f}")

        if max_err > 1e3:
            # Check for scaling
            nonzero = golden != 0
            if np.any(nonzero):
                ratios = C[nonzero] / golden[nonzero]
                median_ratio = np.median(ratios)
                print(f"   *** HUGE ERROR — median(fpga/golden) = {median_ratio:.4f}")
                if abs(median_ratio) > 1.5:
                    import math
                    bits = math.log2(abs(median_ratio))
                    print(f"   *** Scaling factor ~2^{bits:.1f} — fixed-point conversion bug on board")
                elif abs(median_ratio) < 0.01:
                    print(f"   *** Results near zero — board may be returning raw fixed-point integers")
        elif max_err < 1.0:
            print(f"   *** PASS — board produces correct results for 256x256 tiles")

    # 5. Board's built-in verification (summary mode)
    print()
    print("=" * 60)
    print("5. Board's own verification (summary mode, 8x8)")
    print("=" * 60)
    size = max(tile_rows or 4, tile_depth or 4)
    A = rng.standard_normal((size, size)).astype(np.float32)
    B = rng.standard_normal((size, size)).astype(np.float32)
    r = httpx.post(f"{url}/multiply", json={
        "A": A.tolist(),
        "B": B.tolist(),
        "return_mode": "summary",
    }, timeout=30)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR: {data['error']}")
    else:
        print(f"   Board says correct: {data.get('verify', {}).get('correct')}")
        print(f"   Board max error:    {data.get('verify', {}).get('max_err')}")
        print(f"   Stats: {data.get('stats')}")


if __name__ == "__main__":
    main()
