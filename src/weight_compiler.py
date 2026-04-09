#!/usr/bin/env python3
"""Weight-to-metal compiler — PyTorch state_dict to mask-ready binary format."""
import struct, math, random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum


class Precision(Enum):
    FP32 = ("fp32", 32)
    INT8 = ("int8", 8)
    INT4 = ("int4", 4)
    TERNARY = ("ternary", 2)

    def __init__(self, label, bits):
        self.label = label
        self.bits = bits


# Layer type → precision assignment
LAYER_PRECISION = {
    "layernorm": Precision.FP32,
    "rmsnorm": Precision.FP32,
    "embed": Precision.INT8,
    "lm_head": Precision.INT8,
    "q_proj": Precision.INT4,
    "k_proj": Precision.INT4,
    "v_proj": Precision.INT4,
    "o_proj": Precision.INT4,
    "gate_proj": Precision.INT4,
    "up_proj": Precision.INT4,
    "down_proj": Precision.INT4,
    "qkv": Precision.INT4,
    "out": Precision.INT4,
    "ffn_up": Precision.INT4,
    "ffn_down": Precision.INT4,
    "ffn_gate": Precision.INT4,
}


def detect_layer_type(name: str) -> str:
    name_lower = name.lower()
    for pattern in ["layernorm", "rmsnorm"]:
        if pattern in name_lower:
            return pattern
    if "embed" in name_lower or "wte" in name_lower or "wpe" in name_lower:
        return "embed"
    if "lm_head" in name_lower:
        return "lm_head"
    for p in ["q_proj", "k_proj", "v_proj"]:
        if p in name_lower:
            return p.replace("_proj", "")
    if "qkv" in name_lower:
        return "qkv"
    if "o_proj" in name_lower or "output" in name_lower:
        return "o_proj"
    if "gate" in name_lower:
        return "gate_proj"
    if "up_proj" in name_lower or "w_up" in name_lower:
        return "up_proj"
    if "down_proj" in name_lower or "w_down" in name_lower:
        return "down_proj"
    return "ffn_down"


def get_precision(name: str) -> Precision:
    ltype = detect_layer_type(name)
    return LAYER_PRECISION.get(ltype, Precision.INT4)


@dataclass
class LayerInfo:
    name: str
    layer_type: str
    precision: Precision
    shape: Tuple[int, ...]
    num_weights: int
    byte_offset: int = 0
    byte_size: int = 0
    sqnr_db: float = 0.0
    max_error: float = 0.0


@dataclass
class CompiledChip:
    model_name: str
    layers: List[LayerInfo]
    binary: bytes
    layer_table: bytes  # binary index
    total_bytes: int = 0
    die_estimate_mm2: float = 0.0
    compression_ratio: float = 1.0

    def summary(self) -> str:
        lines = [f"Compiled: {self.model_name}", f"Layers: {len(self.layers)}",
                 f"Binary: {self.total_bytes:,} bytes ({self.total_bytes/1e6:.2f} MB)",
                 f"Compression: {self.compression_ratio:.1f}x vs FP32",
                 f"Die estimate: {self.die_estimate_mm2:.1f} mm2", ""]
        by_prec = {}
        for l in self.layers:
            p = l.precision.label
            if p not in by_prec:
                by_prec[p] = {"count": 0, "bytes": 0, "weights": 0}
            by_prec[p]["count"] += 1
            by_prec[p]["bytes"] += l.byte_size
            by_prec[p]["weights"] += l.num_weights
        for p in sorted(by_prec.keys()):
            d = by_prec[p]
            lines.append(f"  {p:8s}: {d['count']:3d} layers, {d['weights']:>12,} weights, {d['bytes']:>10,} bytes")
        return "\n".join(lines)


class WeightCompiler:
    """Compiles neural network weights to mask-ready binary format."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_model(self, name: str, hidden_dim: int = 768, num_layers: int = 24,
                       ff_mult: int = 4, vocab_size: int = 32000,
                       max_per_layer: int = 5000) -> Dict[str, List[float]]:
        """Generate simulated model weights."""
        weights = {}
        for i in range(num_layers):
            p = f"model.layers.{i}"
            n = min(hidden_dim * hidden_dim, max_per_layer)
            weights[f"{p}.self_attn.q_proj.weight"] = self._normal(n, hidden_dim)
            weights[f"{p}.self_attn.k_proj.weight"] = self._normal(n, hidden_dim)
            weights[f"{p}.self_attn.v_proj.weight"] = self._normal(n, hidden_dim)
            weights[f"{p}.self_attn.o_proj.weight"] = self._normal(n, hidden_dim)
            weights[f"{p}.mlp.gate_proj.weight"] = self._normal(min(ff_mult * hidden_dim * hidden_dim, max_per_layer), hidden_dim)
            weights[f"{p}.mlp.up_proj.weight"] = self._normal(min(ff_mult * hidden_dim * hidden_dim, max_per_layer), hidden_dim)
            weights[f"{p}.mlp.down_proj.weight"] = self._normal(min(hidden_dim * ff_mult * hidden_dim, max_per_layer), ff_mult * hidden_dim)
            weights[f"{p}.input_layernorm.weight"] = [1.0 + self.rng.gauss(0, 0.01) for _ in range(hidden_dim)]
            weights[f"{p}.post_attention_layernorm.weight"] = [1.0 + self.rng.gauss(0, 0.01) for _ in range(hidden_dim)]
        weights["model.embed_tokens.weight"] = self._embed(min(vocab_size * hidden_dim, max_per_layer), hidden_dim)
        weights["lm_head.weight"] = self._normal(min(vocab_size * hidden_dim, max_per_layer), hidden_dim)
        weights["model.norm.weight"] = [1.0 + self.rng.gauss(0, 0.01) for _ in range(hidden_dim)]
        return weights

    def _normal(self, n, fan_in):
        std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.1
        return [self.rng.gauss(0, std) for _ in range(n)]

    def _embed(self, n, dim):
        std = math.sqrt(1.0 / dim) * 2
        return [self.rng.gauss(0, std) for _ in range(n)]

    def quantize_fp32(self, weights: List[float]) -> Tuple[List[float], Dict]:
        return weights[:], {"mse": 0, "sqnr_db": float("inf"), "max_error": 0}

    def quantize_int8(self, weights: List[float]) -> Tuple[List[bytes], Dict]:
        wmin, wmax = min(weights), max(weights)
        rng = wmax - wmin
        if rng == 0:
            rng = 1
        qmax = 127
        scale = rng / (2 * qmax)
        quantized = []
        errors = []
        for w in weights:
            q = max(-128, min(127, int(round((w - wmin) / scale - qmax))))
            quantized.append(struct.pack("b", q))
            reconstructed = (q + qmax) * scale + wmin
            errors.append((w - reconstructed) ** 2)
        mse = sum(errors) / len(errors)
        signal = sum(w * w for w in weights) / len(weights)
        sqnr = 10 * math.log10(signal / mse) if mse > 0 else float("inf")
        return quantized, {"mse": mse, "sqnr_db": round(sqnr, 1), "max_error": round(max(abs(e) for e in errors), 8)}

    def quantize_int4(self, weights: List[float]) -> Tuple[List[bytes], Dict]:
        wmin, wmax = min(weights), max(weights)
        rng = wmax - wmin
        if rng == 0:
            rng = 1
        scale = rng / 15.0
        midpoint = (wmax + wmin) / 2
        packed = []
        errors = []
        i = 0
        while i < len(weights):
            vals = []
            for j in range(2):
                if i + j < len(weights):
                    q = max(0, min(15, int(round((weights[i + j] - midpoint) / scale))))
                    vals.append(q)
                    reconstructed = q * scale + midpoint
                    errors.append((weights[i + j] - reconstructed) ** 2)
                else:
                    vals.append(0)
            byte = (vals[0] << 4) | vals[1]
            packed.append(struct.pack("B", byte))
            i += 2
        mse = sum(errors) / len(errors)
        signal = sum(w * w for w in weights) / len(weights)
        sqnr = 10 * math.log10(signal / mse) if mse > 0 else float("inf")
        return packed, {"mse": mse, "sqnr_db": round(sqnr, 1), "max_error": round(max(abs(e) for e in errors), 8)}

    def quantize_ternary(self, weights: List[float]) -> Tuple[List[bytes], Dict]:
        scale = sum(abs(w) for w in weights) / len(weights) + 1e-8
        packed = []
        errors = []
        i = 0
        while i < len(weights):
            nibble = 0
            for j in range(4):
                if i + j < len(weights):
                    w = weights[i + j]
                    if w > scale * 0.5:
                        code = 0b10  # +1
                        reconstructed = scale
                    elif w < -scale * 0.5:
                        code = 0b00  # -1
                        reconstructed = -scale
                    else:
                        code = 0b01  # 0
                        reconstructed = 0
                    errors.append((w - reconstructed) ** 2)
                    nibble |= (code << (j * 2))
            packed.append(struct.pack("B", nibble))
            i += 4
        mse = sum(errors) / len(errors)
        signal = sum(w * w for w in weights) / len(weights)
        sqnr = 10 * math.log10(signal / mse) if mse > 0 else float("inf")
        return packed, {"mse": mse, "sqnr_db": round(sqnr, 1), "max_error": round(max(abs(e) for e in errors), 8)}

    def compile(self, model_name: str, weights: Dict[str, List[float]]) -> CompiledChip:
        binary = bytearray()
        layers = []
        offset = 0

        for name, w in weights.items():
            prec = get_precision(name)
            ltype = detect_layer_type(name)
            info = LayerInfo(name=name, layer_type=ltype, precision=prec,
                           shape=(len(w),), num_weights=len(w), byte_offset=offset)

            if prec == Precision.FP32:
                packed, metrics = self.quantize_fp32(w)
                data = b"".join(struct.pack("<f", v) for v in packed)
            elif prec == Precision.INT8:
                packed, metrics = self.quantize_int8(w)
                data = b"".join(packed)
            elif prec == Precision.INT4:
                packed, metrics = self.quantize_int4(w)
                data = b"".join(packed)
            else:  # TERNARY
                packed, metrics = self.quantize_ternary(w)
                data = b"".join(packed)

            info.byte_size = len(data)
            info.sqnr_db = metrics["sqnr_db"]
            info.max_error = metrics["max_error"]
            info.byte_offset = offset
            binary.extend(data)
            offset += len(data)
            layers.append(info)

        # Layer table: 64 bytes per entry
        # [name:48][offset:4][size:4][precision:1][sqnr:4][padding:3]
        table = bytearray()
        for l in layers:
            name_bytes = l.name.encode("ascii")[:48].ljust(48, b"\x00")
            table.extend(name_bytes)
            table.extend(struct.pack("<I", l.byte_offset))
            table.extend(struct.pack("<I", l.byte_size))
            table.extend(struct.pack("B", l.precision.bits))
            table.extend(struct.pack("<f", l.sqnr_db))
            table.extend(b"\x00" * 3)

        fp32_bytes = sum(l.num_weights * 4 for l in layers)
        compression = fp32_bytes / len(binary) if len(binary) > 0 else 1

        # Die estimation (28nm, ~4 F2 per bit)
        f2_nm2 = 70 ** 2
        die_mm2 = len(binary) * 8 * 4 * f2_nm2 * 1e-12 * 1e6

        return CompiledChip(model_name=model_name, layers=layers,
                          binary=bytes(binary),
                          layer_table=bytes(table), total_bytes=len(binary),
                          die_estimate_mm2=round(die_mm2, 1),
                          compression_ratio=round(compression, 1))


def demo():
    print("=== Frozen Intelligence: Weight-to-Metal Compiler ===\n")

    compiler = WeightCompiler()

    # Scout (1B)
    print("--- Compiling Scout (1B, 512d, 12L) ---")
    w1 = compiler.generate_model("scout-1b", hidden_dim=512, num_layers=12, ff_mult=4, vocab_size=32000)
    chip1 = compiler.compile("scout-1b", w1)
    print(chip1.summary())
    print()

    # Messenger (3B)
    print("--- Compiling Messenger (3B, 768d, 24L) ---")
    w3 = compiler.generate_model("messenger-3b", hidden_dim=768, num_layers=24, ff_mult=4, vocab_size=32000)
    chip3 = compiler.compile("messenger-3b", w3)
    print(chip3.summary())
    print()

    # Sample layer table
    print("--- Layer Table Sample (first 5 layers) ---")
    for l in chip3.layers[:5]:
        print(f"  {l.name:50s} {l.precision.label:8s} {l.num_weights:>8,}w  {l.byte_size:>8,}B  SQNR:{l.sqnr_db:>6.1f}dB")
    print()

    # Precision distribution
    print("--- Precision Distribution (Messenger) ---")
    by_p = {}
    for l in chip3.layers:
        p = l.precision.label
        if p not in by_p:
            by_p[p] = {"layers": 0, "weights": 0, "bytes": 0, "sqnr_sum": 0}
        by_p[p]["layers"] += 1
        by_p[p]["weights"] += l.num_weights
        by_p[p]["bytes"] += l.byte_size
        by_p[p]["sqnr_sum"] += l.sqnr_db
    for p in sorted(by_p.keys()):
        d = by_p[p]
        avg_sqnr = d["sqnr_sum"] / d["layers"] if d["layers"] > 0 else 0
        print(f"  {p:8s}: {d['layers']:3d} layers | {d['weights']:>10,} weights | {d['bytes']:>10,} bytes | avg SQNR {avg_sqnr:.1f}dB")


if __name__ == "__main__":
    demo()
