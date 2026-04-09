#!/usr/bin/env python3
"""TLMM (Table-Lookup MatMul) engine — ternary inference without multipliers.

Based on arXiv:2510.15926 "TeLLMe: Table-Lookup MatMul for Efficient Ternary Inference"
and SuperInstance FPGA Prototype Implementation Guide.
"""
import struct, math
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum


class TernaryWeight(Enum):
    NEG = -1
    ZERO = 0
    POS = 1

    @classmethod
    def encode(cls, value: int) -> int:
        """2-bit encoding: 00=-1, 01=0, 10=+1, 11=reserved"""
        return {cls.NEG.value: 0b00, cls.ZERO.value: 0b01, cls.POS.value: 0b10}.get(value, 0b01)

    @classmethod
    def decode(cls, code: int) -> int:
        return {0b00: cls.NEG.value, 0b01: cls.ZERO.value, 0b10: cls.POS.value}.get(code, 0)


@dataclass
class TLMMConfig:
    activation_bits: int = 4      # 4-bit activation quantization (16 levels)
    accumulator_bits: int = 32    # 32-bit accumulator
    table_size: int = 16          # 16-entry LUT per weight type
    pipeline_depth: int = 3       # 3-stage pipeline
    clock_mhz: int = 500

    @property
    def activation_levels(self) -> int:
        return 1 << self.activation_bits

    @property
    def table_entries(self) -> int:
        return self.activation_levels * 3  # -1, 0, +1


class TLMMTable:
    """Pre-computed lookup table for activation × ternary weight."""

    def __init__(self, config: TLMMConfig):
        self.config = config
        self.table = self._build_table()

    def _build_table(self) -> List[List[int]]:
        """Build 3 tables: for weight = -1, 0, +1."""
        tables = []
        # Dequantization: symmetric around zero
        dequant = []
        for i in range(self.config.activation_levels):
            # Map index to value: -1.0 to +1.0
            val = (i / (self.config.activation_levels - 1)) * 2 - 1
            dequant.append(val)

        for weight in [-1, 0, 1]:
            table = []
            for act_val in dequant:
                product = act_val * weight
                # Scale to accumulator range
                scaled = int(product * (1 << (self.config.accumulator_bits - 8)))
                table.append(scaled)
            tables.append(table)
        return tables

    def lookup(self, activation_idx: int, weight_code: int) -> int:
        """Look up product of activation × weight."""
        weight_val = TernaryWeight.decode(weight_code)
        if weight_val == -1:
            table_idx = 0
        elif weight_val == 0:
            table_idx = 1
        else:  # +1
            table_idx = 2
        return self.table[table_idx][activation_idx]


class TLMMProcessingElement:
    """Single TLMM processing element."""

    def __init__(self, config: TLMMConfig):
        self.config = config
        self.table = TLMMTable(config)
        self.accumulator = 0
        self.valid_out = False

    def step(self, activation_idx: int, weight_code: int, valid_in: bool) -> Tuple[int, bool]:
        """Single computation step."""
        if not valid_in:
            self.valid_out = False
            return self.accumulator, False

        product = self.table.lookup(activation_idx, weight_code)
        self.accumulator += product
        self.valid_out = True
        return self.accumulator, True

    def reset(self):
        self.accumulator = 0
        self.valid_out = False


class TLMMArray:
    """Systolic array of TLMM PEs."""

    def __init__(self, rows: int, cols: int, config: TLMMConfig):
        self.rows = rows
        self.cols = cols
        self.config = config
        self.pes = [[TLMMProcessingElement(config) for _ in range(cols)] for _ in range(rows)]
        self.weights = [[0] * cols for _ in range(rows)]  # 2-bit weight codes

    def load_weights(self, weights: List[List[int]]):
        """Load ternary weights (values -1, 0, +1)."""
        for r in range(self.rows):
            for c in range(self.cols):
                if r < len(weights) and c < len(weights[r]):
                    self.weights[r][c] = TernaryWeight.encode(weights[r][c])

    def compute(self, activations: List[int]) -> List[int]:
        """Compute matrix-vector product: weights × activations."""
        # Reset accumulators
        for row in self.pes:
            for pe in row:
                pe.reset()

        # Systolic flow: activations flow down columns
        results = [0] * self.rows
        for col in range(self.cols):
            act_idx = activations[col] if col < len(activations) else 0
            for row in range(self.rows):
                acc, _ = self.pes[row][col].step(act_idx, self.weights[row][col], True)
                if col == self.cols - 1:  # Last column
                    results[row] = acc
        return results

    def resource_estimate(self) -> Dict:
        """Estimate FPGA resources."""
        # TLMM PE: ~20 LUTs vs traditional MAC: ~200 LUTs
        luts_per_pe = 20
        total_luts = self.rows * self.cols * luts_per_pe
        # Control logic overhead
        total_luts += self.rows * self.cols * 5
        # BRAM for weight storage: 2 bits per weight
        weight_bits = self.rows * self.cols * 2
        weight_bytes = (weight_bits + 7) // 8
        bram36k = math.ceil(weight_bytes / 4500)  # 36Kb BRAM = 4.5KB usable
        return {"luts": total_luts, "bram36k": bram36k, "pe_count": self.rows * self.cols}


class ActivationQuantizer:
    """Quantize FP16/FP32 activations to 4-bit indices."""

    def __init__(self, bits: int = 4, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.levels = 1 << bits

    def quantize(self, activation: float) -> int:
        """Quantize single activation to index."""
        if self.symmetric:
            # Map [-1.0, +1.0] to [0, levels-1]
            clipped = max(-1.0, min(1.0, activation))
            scaled = (clipped + 1.0) / 2.0  # [0, 1]
            idx = int(round(scaled * (self.levels - 1)))
            return max(0, min(self.levels - 1, idx))
        else:
            # Asymmetric quantization (not used in TLMM)
            return int(round(activation * (self.levels - 1) / 2))


def demo():
    print("=== TLMM Engine (Table-Lookup MatMul) ===\n")

    config = TLMMConfig(activation_bits=4, accumulator_bits=32, clock_mhz=500)
    print(f"Config: {config.activation_bits}-bit activation, {config.accumulator_bits}-bit accumulator")
    print(f"Table size: {config.table_entries} entries (16×3)\n")

    # Build table
    table = TLMMTable(config)
    print("--- LUT Values Sample ---")
    for weight, label in [(-1, "-1"), (0, "0"), (1, "+1")]:
        idx = 0 if weight == -1 else (1 if weight == 0 else 2)
        print(f"  Weight {label:2s}: [{table.table[idx][0]:6d}, {table.table[idx][8]:6d}, ..., {table.table[idx][15]:6d}]")
    print()

    # Test PE
    pe = TLMMProcessingElement(config)
    print("--- Processing Element Test ---")
    for act_idx, weight_val in [(0, -1), (8, 0), (15, 1)]:
        weight_code = TernaryWeight.encode(weight_val)
        acc, valid = pe.step(act_idx, weight_code, True)
        print(f"  Activation[{act_idx:2d}] × Weight {weight_val:2d} = {acc:8d} {'✓' if valid else ''}")
    print()

    # Array test
    print("--- 4×4 TLMM Array Test ---")
    array = TLMMArray(4, 4, config)
    weights = [
        [1, -1, 0, 1],
        [0, 1, -1, 0],
        [-1, 0, 1, -1],
        [1, 1, -1, 0],
    ]
    array.load_weights(weights)
    activations = [1, -1, 1, -1]  # Represented as quantized indices
    # Map activations to indices (simplified)
    quant = ActivationQuantizer()
    act_indices = [quant.quantize(a) for a in activations]
    results = array.compute(act_indices)
    print(f"  Weights 4×4, Activations 4×1")
    print(f"  Results: {results}")
    print()

    # Resource estimate
    print("--- Resource Estimation ---")
    for rows, cols in [(4, 4), (16, 16), (64, 64)]:
        arr = TLMMArray(rows, cols, config)
        res = arr.resource_estimate()
        print(f"  {rows:3d}×{cols:<3d}: {res['pe_count']:5d} PEs, {res['luts']:7,d} LUTs, {res['bram36k']} BRAM36K")
    print()

    # Comparison vs traditional MAC
    print("--- TLMM vs Traditional MAC ---")
    print("  Traditional INT8 MAC: ~200 LUTs per PE, 1 DSP per PE")
    print("  TLMM Ternary: ~20 LUTs per PE, 0 DSPs")
    print("  Savings: 90% LUTs, 100% DSPs")
    print("  Power: ~1mW per PE vs ~10mW per MAC")
    print("\n  TLMM enables 10× larger arrays on same FPGA fabric.")


if __name__ == "__main__":
    demo()
