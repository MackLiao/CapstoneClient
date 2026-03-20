from pynq import Overlay, MMIO, allocate
import numpy as np

N_ROWS = 8
N_COLS = 8
TILE_DEPTH = 128

# Maximum matrix dimension supported (for buffer pre-allocation)
MAX_DIM = 1024

# -------------------------------------------------------------------
# Register maps
# -------------------------------------------------------------------
# s_axi_control  — mapped at 0x4000_0000, 64K range
CTRL_BASE  = 0x40000000
CTRL_RANGE = 0x10000
AP_CTRL = 0x00
REG_M   = 0x10
REG_N   = 0x18
REG_K   = 0x20

# s_axi_control_r — mapped at 0x4001_0000, 64K range
CTRL_R_BASE  = 0x40010000
CTRL_R_RANGE = 0x10000
MEM_A_ADDR_LO = 0x10
MEM_A_ADDR_HI = 0x14
MEM_B_ADDR_LO = 0x1C
MEM_B_ADDR_HI = 0x20
MEM_C_ADDR_LO = 0x28
MEM_C_ADDR_HI = 0x2C


class MatrixAccelerator:
    # Fixed-point config for 16-bit data (ap_int<16>).
    # Hardware does raw MAC (no post-multiply shift), so the accumulator
    # has 2*frac_bits fractional bits → divide by scale² on output.
    #
    # Tradeoff: more frac_bits = better precision, smaller input range.
    #   frac_bits=14  → Q1.14, range [-2, ~2),      step ≈ 0.00006
    #   frac_bits=12  → Q3.12, range [-8, ~8),       step ≈ 0.00024
    #   frac_bits=8   → Q7.8,  range [-128, ~128),   step ≈ 0.004
    DEFAULT_FRAC_BITS = 14

    def __init__(self, bitstream_path, frac_bits=None):
        self.frac_bits = frac_bits if frac_bits is not None else self.DEFAULT_FRAC_BITS
        self.scale    = 1 << self.frac_bits
        self.scale_sq = 1 << (2 * self.frac_bits)

        self.overlay = Overlay(bitstream_path)

        # MMIO handles for the two AXI-Lite interfaces
        self.ctrl   = MMIO(CTRL_BASE, CTRL_RANGE)
        self.ctrl_r = MMIO(CTRL_R_BASE, CTRL_R_RANGE)

        # Contiguous buffers for full matrices (pre-allocate for MAX_DIM)
        #   mem_A (data_t* = ap_int<16>*):  up to MAX_DIM × MAX_DIM elements
        #   mem_B (data_t* = ap_int<16>*):  up to MAX_DIM × MAX_DIM elements
        #   mem_C (acc_t*  = ap_int<48>*):  up to MAX_DIM × MAX_DIM elements (64-bit on bus)
        self.buf_A = allocate(shape=(MAX_DIM * MAX_DIM,), dtype=np.int16)
        self.buf_B = allocate(shape=(MAX_DIM * MAX_DIM,), dtype=np.int16)
        self.buf_C = allocate(shape=(MAX_DIM * MAX_DIM,), dtype=np.int64)

        # Write buffer physical addresses once (they don't change between calls)
        self._write_addr(MEM_A_ADDR_LO, self.buf_A.physical_address)
        self._write_addr(MEM_B_ADDR_LO, self.buf_B.physical_address)
        self._write_addr(MEM_C_ADDR_LO, self.buf_C.physical_address)

    def _write_addr(self, lo_offset, phys_addr):
        """Write a 64-bit physical address to control_r as two 32-bit halves."""
        self.ctrl_r.write(lo_offset, int(phys_addr) & 0xFFFFFFFF)
        self.ctrl_r.write(lo_offset + 4, (int(phys_addr) >> 32) & 0xFFFFFFFF)

    def _start_and_wait(self):
        """Assert AP_START and poll until AP_DONE."""
        self.ctrl.write(AP_CTRL, 0x01)
        while True:
            if self.ctrl.read(AP_CTRL) & 0x02:  # bit 1 = ap_done
                break

    def _to_fixed(self, matrix):
        """Convert to int16. Integers pass through; floats use configurable Q format."""
        if np.issubdtype(matrix.dtype, np.integer):
            return matrix.astype(np.int16)
        return np.clip(
            np.round(matrix * self.scale), -32768, 32767
        ).astype(np.int16)

    def multiply(self, mat_A, mat_B):
        """Multiply two matrices using the FPGA accelerator.

        Matrices must have dimensions divisible by 8 (N_ROWS/N_COLS) and
        inner dimension (K) divisible by 128 (TILE_DEPTH).

        For float inputs, uses configurable fixed-point (default Q1.14, range [-2, ~2)).
        For integer inputs, passes values directly as int16.
        """
        M, K_a = mat_A.shape
        K_b, N = mat_B.shape
        assert K_a == K_b, f"Inner dimensions must match: {K_a} vs {K_b}"
        K = K_a
        assert M % N_ROWS == 0, f"M ({M}) must be divisible by {N_ROWS}"
        assert N % N_COLS == 0, f"N ({N}) must be divisible by {N_COLS}"
        assert K % TILE_DEPTH == 0, f"K ({K}) must be divisible by {TILE_DEPTH}"
        assert M <= MAX_DIM and N <= MAX_DIM and K <= MAX_DIM, \
            f"Dimensions must be <= {MAX_DIM}"

        is_float = np.issubdtype(mat_A.dtype, np.floating)
        mat_A_fixed = self._to_fixed(mat_A)
        mat_B_fixed = self._to_fixed(mat_B)

        # Copy full matrices into contiguous DMA buffers (row-major)
        self.buf_A[:M * K] = mat_A_fixed.reshape(-1)
        self.buf_B[:K * N] = mat_B_fixed.reshape(-1)

        # Write dimensions
        self.ctrl.write(REG_M, M)
        self.ctrl.write(REG_N, N)
        self.ctrl.write(REG_K, K)

        # Single kernel invocation
        self._start_and_wait()

        # Read result
        raw_C = self.buf_C[:M * N].reshape(M, N).copy()

        # Sign-extend 48-bit accumulator values to 64-bit.
        # The hardware writes 48-bit signed results into a 64-bit buffer,
        # leaving bits 48-63 as zero. Shift left 16 to put the 48-bit sign
        # bit at the int64 sign position, then arithmetic right-shift back.
        raw_C = (raw_C << 16) >> 16

        if is_float:
            return raw_C.astype(np.float32) / self.scale_sq
        else:
            return raw_C

    def __del__(self):
        for buf in [self.buf_A, self.buf_B, self.buf_C]:
            buf.freebuffer()
