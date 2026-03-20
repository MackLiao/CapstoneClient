"""
Flask server for the systolic array matrix multiplication accelerator.
Runs on the PYNQ board.

Usage:
    python3 app.py [--bitstream PATH] [--frac-bits N] [--port PORT]
"""

import argparse
import time
import numpy as np
from flask import Flask, request, jsonify
from matrixMultiplicationAccelerator import MatrixAccelerator, N_ROWS, N_COLS, TILE_DEPTH

app = Flask(__name__)
accel = None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "tile_rows": N_ROWS,
        "tile_cols": N_COLS,
        "tile_depth": TILE_DEPTH,
        "frac_bits": accel.frac_bits,
    })


@app.route('/multiply', methods=['POST'])
def multiply():
    data = request.get_json(silent=True)

    request_id = None
    if isinstance(data, dict):
        request_id = data.get("request_id")

    def make_error(message: str, code: int = 400):
        return jsonify({
            "request_id": request_id,
            "status": "error",
            "mode": None,
            "shape": None,
            "elapsed_sec": None,
            "result": None,
            "stats": None,
            "verify": None,
            "error": message,
        }), code

    if not isinstance(data, dict):
        return make_error("Request body must be valid JSON")

    if 'A' not in data or 'B' not in data:
        return make_error("Request must include 'A' and 'B' matrices")

    return_mode = data.get("return_mode", "auto")
    if return_mode not in ("auto", "full", "summary"):
        return make_error("return_mode must be one of: auto, full, summary")

    try:
        mat_a = np.array(data['A'], dtype=np.float32)
        mat_b = np.array(data['B'], dtype=np.float32)
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

    if M % N_ROWS != 0:
        return make_error(f"Rows of A ({M}) must be divisible by {N_ROWS}")

    if N % N_COLS != 0:
        return make_error(f"Cols of B ({N}) must be divisible by {N_COLS}")

    if K_a % TILE_DEPTH != 0:
        return make_error(f"Inner dimension ({K_a}) must be divisible by {TILE_DEPTH}")

    try:
        t0 = time.time()
        result = accel.multiply(mat_a, mat_b)
        elapsed = round(time.time() - t0, 6)

        stats = {
            "sum": float(np.sum(result)),
            "max": float(np.max(result)),
            "min": float(np.min(result)),
        }

        auto_should_return_full = (M <= 256 and N <= 256 and K_a <= 256)

        if return_mode == "full" or (return_mode == "auto" and auto_should_return_full):
            return jsonify({
                "request_id": request_id,
                "status": "ok",
                "mode": "full",
                "shape": [int(M), int(N)],
                "elapsed_sec": elapsed,
                "result": {
                    "C": result.tolist()
                },
                "stats": stats,
                "verify": None,
                "error": None,
            })

        # Summary mode: verify on board, return stats only
        golden = mat_a @ mat_b
        diff = np.abs(result - golden)
        worst = np.unravel_index(np.argmax(diff), diff.shape)

        verify = {
            "correct": bool(np.allclose(result, golden, atol=1e-1)),
            "max_err": float(diff[worst]),
            "worst_index": [int(worst[0]), int(worst[1])],
            "server_value": float(result[worst]),
            "golden_value": float(golden[worst]),
        }

        return jsonify({
            "request_id": request_id,
            "status": "ok",
            "mode": "summary",
            "shape": [int(M), int(N)],
            "elapsed_sec": elapsed,
            "result": None,
            "stats": stats,
            "verify": verify,
            "error": None,
        })

    except Exception as e:
        return jsonify({
            "request_id": request_id,
            "status": "error",
            "mode": None,
            "shape": None,
            "elapsed_sec": None,
            "result": None,
            "stats": None,
            "verify": None,
            "error": str(e),
        }), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bitstream', default='design_1_wrapper.bit')
    parser.add_argument('--frac-bits', type=int, default=None,
                        help='Fixed-point fractional bits')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    print(f"Loading bitstream: {args.bitstream}")
    accel = MatrixAccelerator(args.bitstream, frac_bits=args.frac_bits)
    print(
        f"Accelerator ready ({N_ROWS}x{N_COLS} array, "
        f"TILE_DEPTH={TILE_DEPTH}, frac_bits={accel.frac_bits})"
    )

    app.run(host='0.0.0.0', port=args.port)
