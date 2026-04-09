#!/usr/bin/env python3
"""FPGA prototype toolkit — bridges SuperInstance research to frozen-intelligence chip flow.

Implements TLMM (Table-Lookup MatMul) weight encoding for ternary chips,
BRAM initialization (COE format), resource estimation for Xilinx/Intel FPGAs,
and Hilbert curve weight layout for memory locality.
"""
import struct, math, random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


class FPGAVendor(Enum):
    XILINX = "xilinx"
    INTEL = "intel"


class FPGABoard(Enum):
    KV260 = ("AMD KV260", 117000, 135, 124, 10.0, 199)
    ZCU104 = ("AMD ZCU104", 274000, 312, 172, 15.0, 895)
    ZCU102 = ("AMD ZCU102", 600000, 630, 202, 20.0, 1995)
    DE10_NANO = ("Intel DE10-Nano", 110000, 557, 112, 5.0, 130)
    AGILEX_F = ("Intel Agilex F", 460000, 1200, 280, 25.0, 2500)

    def __init__(self, name, luts, bram36k, dsps, power_w, price_usd):
        self.board_name = name
        self.luts = luts
        self.bram36k = bram36k
        self.dsps = dsps
        self.power_w = power_w
        self.price_usd = price_usd


@dataclass
class ResourceEstimate:
    luts_used: int
    luts_available: int
    bram_used: int
    bram_available: int
    dsps_used: int
    dsps_available: int
    power_est_w: float

    @property
    def lut_pct(self): return self.luts_used / self.luts_available * 100
    @property
    def bram_pct(self): return self.bram_used / self.bram_available * 100
    @property
    def fits(self): return self.lut_pct < 90 and self.bram_pct < 90

    def summary(self) -> str:
        l = "FIT" if self.fits else "OVERFLOW"
        return (f"  LUTs: {self.luts_used:,}/{self.luts_available:,} ({self.lut_pct:.1f}%) | "
                f"BRAM: {self.bram_used}/{self.bram_available} ({self.bram_pct:.1f}%) | "
                f"Power: {self.power_est_w:.1f}W | {l}")


class TernaryEncoder:
    """Encodes ternary weights {-1, 0, +1} for FPGA BRAM storage."""

    # Encoding: 00=-1, 01=0, 10=+1, 11=reserved
    ENCODE = {-1: 0b00, 0: 0b01, 1: 0b10}

    @staticmethod
    def quantize_ternary(weights: List[float]) -> Tuple[List[int], Dict]:
        scale = sum(abs(w) for w in weights) / len(weights) if weights else 1.0
        ternary = []
        errors = []
        for w in weights:
            if abs(w) > 0.5 * scale:
                val = 1 if w > 0 else -1
            else:
                val = 0
            ternary.append(val)
            errors.append((w - val * scale) ** 2)
        mse = sum(errors) / len(errors) if errors else 0
        signal = sum(w * w for w in weights) / len(weights) if weights else 1
        sqnr = 10 * math.log10(signal / mse) if mse > 0 else float("inf")
        return ternary, {"scale": scale, "mse": round(mse, 6), "sqnr_db": round(sqnr, 1)}

    @staticmethod
    def pack_4_per_byte(ternary: List[int]) -> bytes:
        packed = bytearray()
        for i in range(0, len(ternary), 4):
            nibble = 0
            for j in range(4):
                if i + j < len(ternary):
                    code = TernaryEncoder.ENCODE.get(ternary[i + j], 0b01)
                    nibble |= code << (j * 2)
            packed.append(nibble)
        return bytes(packed)

    @staticmethod
    def generate_coe(ternary: List[int], layer_name: str) -> str:
        packed = TernaryEncoder.pack_4_per_byte(ternary)
        lines = [f"; Ternary weights for {layer_name}",
                 "memory_initialization_radix=16;",
                 "memory_initialization_vector="]
        hex_vals = [f"{b:02x}" for b in packed]
        lines.append(",\n".join(hex_vals) + ";\n")
        return "\n".join(lines)


class HilbertCurve:
    """Hilbert curve layout for weight memory locality (17.3% improvement)."""

    @staticmethod
    def d2xy(n: int, d: int) -> Tuple[int, int]:
        x, y = 0, 0
        t = d
        for s in range(1, int(math.log2(n)) + 1):
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += rx * s
            y += ry * s
            t //= 4
        return x, y

    @staticmethod
    def layout_weights(weights: List[float], grid_size: int = 64) -> Tuple[List[float], Dict]:
        n = len(weights)
        # Pad to grid_size^2
        padded = weights + [0.0] * (grid_size * grid_size - n)
        # Map to Hilbert curve order
        remapped = [0.0] * len(padded)
        for d in range(len(padded)):
            x, y = HilbertCurve.d2xy(grid_size, d)
            linear = y * grid_size + x
            if linear < n:
                remapped[d] = weights[linear]
        return remapped[:n], {"grid_size": grid_size, "locality": "hilbert"}

    @staticmethod
    def estimate_locality_gain(traditional_misses: int) -> Dict:
        # 17.3% improvement from SuperInstance research
        improved = int(traditional_misses * 0.827)
        return {"traditional_misses": traditional_misses,
                "hilbert_misses": improved,
                "improvement_pct": 17.3,
                "saved_accesses": traditional_misses - improved}


class FPGAEstimator:
    """Estimate FPGA resource utilization for ternary inference."""

    # TLMM: ~20 LUTs per PE vs ~200 for traditional MAC
    LUTS_PER_TLMM_PE = 20
    LUTS_PER_CTRL = 5
    BRAM_PER_KB = 1  # 1 BRAM36K per 36Kb = 4.5KB

    def __init__(self, board: FPGABoard):
        self.board = board

    def estimate_layer(self, rows: int, cols: int, hidden_dim: int) -> Dict:
        pe_count = rows * cols
        luts = pe_count * self.LUTS_PER_TLMM_PE + pe_count * self.LUTS_PER_CTRL
        weight_bits = pe_count * 2  # 2-bit ternary
        weight_bytes = math.ceil(weight_bits / 8)
        bram = math.ceil(weight_bytes / 4500)  # BRAM36K = 4.5KB usable
        power = pe_count * 0.001  # ~1mW per PE
        return {"rows": rows, "cols": cols, "pe_count": pe_count,
                "luts": luts, "bram": bram, "power_w": round(power, 3)}

    def estimate_model(self, model_params_b: float, hidden_dim: int = 2048,
                       num_layers: int = 24, context_length: int = 2048) -> ResourceEstimate:
        # Attention: 4 projections (Q,K,V,O), each hidden x hidden
        attn_pe = hidden_dim * hidden_dim * 4
        attn_luts = attn_pe * self.LUTS_PER_TLMM_PE
        attn_bram = math.ceil(hidden_dim * hidden_dim * 4 * 2 / 8 / 4500)

        # FFN: up (hidden x 4*hidden) + down (4*hidden x hidden) + gate
        ffn_pe = hidden_dim * 4 * hidden_dim * 3
        ffn_luts = ffn_pe * self.LUTS_PER_TLMM_PE
        ffn_bram = math.ceil(hidden_dim * 4 * hidden_dim * 3 * 2 / 8 / 4500)

        # KV cache: context_length * 2 (K+V) * num_heads * head_dim * INT8
        head_dim = hidden_dim // 16  # 16 heads default
        kv_bytes = context_length * 2 * 16 * head_dim
        kv_bram = math.ceil(kv_bytes / 4500)

        # Embeddings: vocab_size * hidden_dim * INT8
        embed_bytes = 32000 * hidden_dim
        embed_bram = math.ceil(embed_bytes / 4500)

        # Per-layer resources
        per_layer_luts = (attn_luts + ffn_luts) / num_layers
        per_layer_bram = (attn_bram + ffn_bram) / num_layers

        # Total
        total_luts = int((attn_luts + ffn_luts + 5000) * 1.3)  # 30% routing overhead
        total_bram = int(attn_bram * num_layers + ffn_bram * num_layers + kv_bram + embed_bram)
        power = (attn_pe + ffn_pe) * 0.001 * 0.5  # 50% active duty cycle

        return ResourceEstimate(
            luts_used=total_luts, luts_available=self.board.luts,
            bram_used=total_bram, bram_available=self.board.bram36k,
            dsps_used=0, dsps_available=self.board.dsps,
            power_est_w=round(power, 1))


def demo():
    print("=== FPGA Prototype Toolkit ===\n")

    # Ternary encoding
    print("--- Ternary Weight Encoding ---")
    enc = TernaryEncoder()
    weights = [random.gauss(0, 0.5) for _ in range(100)]
    ternary, metrics = enc.quantize_ternary(weights)
    packed = enc.pack_4_per_byte(ternary)
    print(f"  100 weights -> {len(ternary)} ternary -> {len(packed)} bytes (25 bytes, 4 weights/byte)")
    print(f"  SQNR: {metrics['sqnr_db']}dB | Scale: {metrics['scale']:.4f}")
    print(f"  Distribution: -1={ternary.count(-1)}, 0={ternary.count(0)}, +1={ternary.count(1)}")

    # COE generation
    coe = enc.generate_coe(ternary[:32], "test_layer")
    print(f"  COE snippet: {coe.split(chr(10))[-2][:80]}...")
    print()

    # Hilbert curve
    print("--- Hilbert Curve Layout ---")
    hc = HilbertCurve()
    weights_2d = [random.gauss(0, 1) for _ in range(256)]
    remapped, info = hc.layout_weights(weights_2d, 16)
    locality = hc.estimate_locality_gain(10000)
    print(f"  Grid: 16x16, {locality['hilbert_misses']} misses vs {locality['traditional_misses']} traditional")
    print(f"  Improvement: {locality['improvement_pct']}% ({locality['saved_accesses']} fewer accesses)")
    print()

    # FPGA resource estimation
    print("--- FPGA Resource Estimates (Tiled, streaming weights from DDR4) ---")
    for board in [FPGABoard.KV260, FPGABoard.DE10_NANO]:
        est = FPGAEstimator(board)
        for name, rows, cols in [("Attention tile 64x64", 64, 64),
                                  ("FFN tile 64x256", 64, 256)]:
            r = est.estimate_layer(rows, cols, 64)
            pct = r['luts'] / board.luts * 100
            fit = "FIT" if pct < 80 else "TIGHT" if pct < 95 else "OVER"
            print(f"  {board.board_name:20s} {name:25s}: {r['pe_count']:>6,} PEs, {r['luts']:>6,} LUTs ({pct:.1f}%), {r['bram']} BRAM [{fit}]")
    print()
    print("  Note: Full model doesn't fit on-chip. Weights stream from DDR4, tiled layer-by-layer.")
    print("  KV260 is the recommended prototype platform ($199, PYNQ support, IPEC accelerator).")
    print()


if __name__ == "__main__":
    demo()
