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
        tile_rows = health.get("tile_rows", 8)
        tile_cols = health.get("tile_cols", 8)
        tile_depth = health.get("tile_depth", 128)
        frac_bits = health.get("frac_bits")
        print(f"   Hardware dims: rows={tile_rows}, cols={tile_cols}, depth={tile_depth}")
        print(f"   frac_bits={frac_bits}")
        if frac_bits and frac_bits >= 14:
            input_range = 2 ** (16 - 1 - frac_bits)
            print(f"   *** Input range: [-{input_range}, ~{input_range})")
            print(f"   *** standard_normal values often exceed this — consider --frac-bits 10")
    except Exception as e:
        print(f"   FAILED: {e}")
        sys.exit(1)

    # Use dimensions compatible with hardware (all must be divisible by
    # tile_rows, tile_cols, AND tile_depth for the inner dimension K)
    # Smallest valid square size: lcm of all three constraints
    from math import lcm
    min_k = tile_depth  # inner dim must be divisible by tile_depth
    min_m = tile_rows   # rows must be divisible by tile_rows
    min_n = tile_cols   # cols must be divisible by tile_cols
    # For square tests, use a size that satisfies all three
    test_dim = lcm(min_m, min_n, min_k)  # 128 for 8,8,128

    # 2. Identity test: I @ I = I
    print()
    print("=" * 60)
    print(f"2. Identity test ({test_dim}x{test_dim}): I @ I should = I")
    print("=" * 60)
    I_mat = np.eye(test_dim, dtype=np.float32)
    r = httpx.post(f"{url}/multiply", json={
        "A": I_mat.tolist(),
        "B": I_mat.tolist(),
        "return_mode": "full",
    }, timeout=60)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR from board: {data['error']}")
    else:
        C = np.array(data["result"]["C"])
        print(f"   Result shape: {C.shape}")
        print(f"   Diagonal sample: {C[0,0]:.6f}, {C[1,1]:.6f}, {C[2,2]:.6f}")
        print(f"   Off-diagonal sample: {C[0,1]:.6f}, {C[1,0]:.6f}")
        err = np.max(np.abs(C - I_mat))
        print(f"   Max error vs identity: {err:.6e}")
        print(f"   {'PASS' if err < 0.1 else '*** FAIL'}")

    # 3. Simple known multiplication
    print()
    print("=" * 60)
    print(f"3. Known multiplication: ones({test_dim},{test_dim}) @ ones({test_dim},{test_dim})")
    print(f"   Expected: every element = {test_dim}.0")
    print("=" * 60)
    A = np.ones((test_dim, test_dim), dtype=np.float32)
    B = np.ones((test_dim, test_dim), dtype=np.float32)
    r = httpx.post(f"{url}/multiply", json={
        "A": A.tolist(),
        "B": B.tolist(),
        "return_mode": "full",
    }, timeout=60)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR from board: {data['error']}")
    else:
        C = np.array(data["result"]["C"])
        expected = float(test_dim)
        print(f"   Result[0,0] = {C[0,0]:.6f}  (expected {expected})")
        err = np.max(np.abs(C - expected))
        print(f"   Max error: {err:.6e}")
        if abs(C[0, 0] - expected) > 1:
            ratio = C[0, 0] / expected
            print(f"   *** SCALING ISSUE: result/expected = {ratio:.2f}")
        else:
            print(f"   {'PASS' if err < 1.0 else '*** FAIL'}")

    # 4. Random tile-sized test (matches what orchestrator sends)
    print()
    print("=" * 60)
    print("4. Random tile test (256x256) — same size orchestrator sends")
    print("=" * 60)
    rng = np.random.default_rng(42)
    # Clamp inputs to the safe fixed-point range to isolate precision from clipping
    if frac_bits:
        safe_range = 2 ** (16 - 1 - frac_bits) * 0.9
    else:
        safe_range = 1.5
    A = np.clip(rng.standard_normal((256, 256)), -safe_range, safe_range).astype(np.float32)
    B = np.clip(rng.standard_normal((256, 256)), -safe_range, safe_range).astype(np.float32)
    golden = A.astype(np.float64) @ B.astype(np.float64)

    r = httpx.post(f"{url}/multiply", json={
        "A": A.tolist(),
        "B": B.tolist(),
        "return_mode": "full",
    }, timeout=120)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR from board: {data['error']}")
    else:
        C = np.array(data["result"]["C"], dtype=np.float64)
        print(f"   Result shape: {C.shape}")
        diff = np.abs(C - golden)
        max_err = np.max(diff)
        mean_err = np.mean(diff)
        print(f"   Max abs error:  {max_err:.6e}")
        print(f"   Mean abs error: {mean_err:.6e}")
        print(f"   Golden[0,0]={golden[0,0]:.4f}  FPGA[0,0]={C[0,0]:.4f}")
        if max_err < 1.0:
            print(f"   PASS")
        elif max_err < 10.0:
            print(f"   MARGINAL — precision limited by frac_bits={frac_bits}")
        else:
            print(f"   *** FAIL — error too large")

    # 5. Random test WITHOUT clamping (shows clipping impact)
    print()
    print("=" * 60)
    print("5. Random test (256x256, unclamped standard_normal)")
    print("=" * 60)
    A_raw = rng.standard_normal((256, 256)).astype(np.float32)
    B_raw = rng.standard_normal((256, 256)).astype(np.float32)
    golden_raw = A_raw.astype(np.float64) @ B_raw.astype(np.float64)

    r = httpx.post(f"{url}/multiply", json={
        "A": A_raw.tolist(),
        "B": B_raw.tolist(),
        "return_mode": "full",
    }, timeout=120)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR from board: {data['error']}")
    else:
        C = np.array(data["result"]["C"], dtype=np.float64)
        diff = np.abs(C - golden_raw)
        max_err = np.max(diff)
        mean_err = np.mean(diff)
        print(f"   Max abs error:  {max_err:.6e}")
        print(f"   Mean abs error: {mean_err:.6e}")
        if frac_bits and frac_bits >= 13:
            clipped = np.sum(np.abs(A_raw) > safe_range) + np.sum(np.abs(B_raw) > safe_range)
            total = A_raw.size + B_raw.size
            print(f"   Values clipped: {clipped}/{total} ({100*clipped/total:.1f}%)")
            print(f"   *** High error expected — frac_bits={frac_bits} clips values outside [-{safe_range:.1f}, {safe_range:.1f}]")
            print(f"   *** Recommend: restart board with --frac-bits 10 (range [-32, 32))")

    # 6. Board's own verification (summary mode)
    print()
    print("=" * 60)
    print(f"6. Board verification (summary mode, {test_dim}x{test_dim})")
    print("=" * 60)
    A = np.clip(rng.standard_normal((test_dim, test_dim)), -safe_range, safe_range).astype(np.float32)
    B = np.clip(rng.standard_normal((test_dim, test_dim)), -safe_range, safe_range).astype(np.float32)
    r = httpx.post(f"{url}/multiply", json={
        "A": A.tolist(),
        "B": B.tolist(),
        "return_mode": "summary",
    }, timeout=60)
    data = r.json()
    if data["status"] == "error":
        print(f"   ERROR: {data['error']}")
    else:
        print(f"   Board says correct: {data.get('verify', {}).get('correct')}")
        print(f"   Board max error:    {data.get('verify', {}).get('max_err')}")
        print(f"   Stats: {data.get('stats')}")


if __name__ == "__main__":
    main()
