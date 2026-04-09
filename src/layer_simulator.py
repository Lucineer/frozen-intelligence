#!/usr/bin/env python3
"""Neural network layer simulator for weight-locked chips.

Simulates transformer layers with mixed precision (FP32/LN, INT8/Embed, INT4/Attn+FFN),
quantization effects, and throughput estimation on mask-locked hardware.
"""
import math, struct, random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class Precision(Enum):
    FP32 = ("fp32", 4)
    FP16 = ("fp16", 2)
    INT8 = ("int8", 1)
    INT4 = ("int4", 0.5)
    BINARY = ("binary", 0.125)

    def __init__(self, label, bytes_per_elem):
        self.label = label
        self.bytes_per_elem = bytes_per_elem


@dataclass
class LayerConfig:
    name: str
    d_model: int
    d_ff: int
    n_heads: int
    precision: Precision = Precision.INT4
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5


@dataclass
class ChipConfig:
    name: str
    mac_units: int
    clock_mhz: int
    bandwidth_gbps: float
    power_w: float
    sram_kb: int


# Vessel class configs
VESSEL_CONFIGS = {
    "scout": ChipConfig("Scout", mac_units=64, clock_mhz=200, bandwidth_gbps=1.6, power_w=0.8, sram_kb=256),
    "messenger": ChipConfig("Messenger", mac_units=256, clock_mhz=400, bandwidth_gbps=6.4, power_w=2.5, sram_kb=1024),
    "navigator": ChipConfig("Navigator", mac_units=1024, clock_mhz=500, bandwidth_gbps=12.8, power_w=5.0, sram_kb=4096),
    "captain": ChipConfig("Captain", mac_units=4096, clock_mhz=600, bandwidth_gbps=25.6, power_w=10.0, sram_kb=16384),
}


class Quantizer:
    """Symmetric quantization."""

    @staticmethod
    def quantize(values: List[float], bits: int) -> Tuple[List[int], float]:
        max_val = max(abs(v) for v in values) or 1.0
        scale = max_val / ((1 << (bits - 1)) - 1)
        quantized = [round(v / scale) for v in values]
        return quantized, scale

    @staticmethod
    def dequantize(values: List[int], scale: float) -> List[float]:
        return [v * scale for v in values]

    @staticmethod
    def quantize_error(values: List[float], bits: int) -> Dict:
        quantized, scale = Quantizer.quantize(values, bits)
        dequant = Quantizer.dequantize(quantized, scale)
        errors = [abs(o - q) for o, q in zip(values, dequant)]
        mse = sum(e * e for e in errors) / len(errors)
        max_err = max(errors)
        snr = 10 * math.log10(sum(v * v for v in values) / (mse * len(values))) if mse > 0 else float('inf')
        return {"mse": round(mse, 6), "max_error": round(max_err, 6),
                "snr_db": round(snr, 2), "bits": bits}


class AttentionHead:
    """Single attention head simulation."""

    def __init__(self, d_head: int, precision: Precision = Precision.INT4):
        self.d_head = d_head
        self.precision = precision
        self.Q = [random.gauss(0, 0.1) for _ in range(d_head)]
        self.K = [random.gauss(0, 0.1) for _ in range(d_head)]
        self.V = [random.gauss(0, 0.1) for _ in range(d_head)]

    def dot_product(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def attention_score(self) -> float:
        return self.dot_product(self.Q, self.K) / math.sqrt(self.d_head)

    def softmax(self, scores: List[float]) -> List[float]:
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def forward(self, kv_cache: Optional[List[float]] = None) -> Dict:
        score = self.attention_score()
        weights = self.softmax([score, score * 0.8])
        output = [self.V[i] * weights[0] + (kv_cache[i] if kv_cache else self.V[i]) * weights[1]
                  for i in range(self.d_head)]
        return {"score": round(score, 4), "weights": [round(w, 4) for w in weights],
                "output_norm": round(math.sqrt(sum(o*o for o in output)), 4)}


class TransformerLayer:
    """Full transformer layer with mixed precision."""

    def __init__(self, config: LayerConfig):
        self.config = config
        self.heads = [AttentionHead(config.d_model // config.n_heads, config.precision)
                      for _ in range(config.n_heads)]

    @property
    def param_count(self) -> int:
        c = self.config
        # Attention: 4 * d_model^2 (Q, K, V, O), LayerNorm: 2 * d_model
        attn = 4 * c.d_model * c.d_model + 2 * c.d_model
        # FFN: 3 * d_model * d_ff, LayerNorm: 2 * d_model
        ffn = 3 * c.d_model * c.d_ff + 2 * c.d_model
        return attn + ffn

    def weight_size_bytes(self) -> Dict:
        c = self.config
        attn = 4 * c.d_model * c.d_model + 2 * c.d_model
        ffn = 3 * c.d_model * c.d_ff + 2 * c.d_model
        return {
            "attention_bytes": int(attn * c.precision.bytes_per_elem),
            "ffn_bytes": int(ffn * c.precision.bytes_per_elem),
            "total_bytes": int((attn + ffn) * c.precision.bytes_per_elem),
        }

    def compute_cycles(self, vessel: ChipConfig) -> Dict:
        c = self.config
        macs_per_head = c.d_model * c.d_model  # simplified
        total_macs = self.heads[0].d_head * c.d_model * c.n_heads * 2  # QK + AV
        ffn_macs = c.d_model * c.d_ff * 3  # 3 linear layers

        attn_cycles = math.ceil(total_macs / vessel.mac_units)
        ffn_cycles = math.ceil(ffn_macs / vessel.mac_units)
        total_cycles = attn_cycles + ffn_cycles

        return {
            "attn_cycles": attn_cycles,
            "ffn_cycles": ffn_cycles,
            "total_cycles": total_cycles,
            "time_us": round(total_cycles / vessel.clock_mhz, 2),
            "tokens_per_sec": round(vessel.clock_mhz / total_cycles, 1),
        }

    def forward(self, input_vec: List[float]) -> Dict:
        results = [h.forward() for h in self.heads]
        output = [sum(r["output_norm"] for r in results) / len(results)]
        return {"output": output, "heads": len(results)}


class ModelSimulator:
    """Full model simulation across vessel classes."""

    def __init__(self, name: str, n_layers: int, d_model: int, d_ff: int,
                 n_heads: int, vocab_size: int = 32000):
        self.name = name
        self.config = LayerConfig(name=name, d_model=d_model, d_ff=d_ff, n_heads=n_heads)
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.layers = [TransformerLayer(self.config) for _ in range(n_layers)]

    @property
    def total_params(self) -> int:
        return sum(l.param_count for l in self.layers) + self.vocab_size * self.config.d_model

    def estimate_size_mb(self, precision: Precision = Precision.INT4) -> float:
        embedding = self.vocab_size * self.config.d_model * precision.bytes_per_elem
        layer_params = sum(l.param_count for l in self.layers) * precision.bytes_per_elem
        return (embedding + layer_params) / (1024 * 1024)

    def throughput(self, vessel: str = "scout") -> Dict:
        vc = VESSEL_CONFIGS[vessel]
        layer_cycles = self.layers[0].compute_cycles(vc)
        total_cycles = layer_cycles["total_cycles"] * self.n_layers
        tps = vc.clock_mhz * 1e6 / total_cycles

        return {
            "model": self.name,
            "vessel": vessel,
            "params_M": round(self.total_params / 1e6, 1),
            "size_MB": round(self.estimate_size_mb(), 2),
            "tokens_per_sec": round(tps, 1),
            "latency_ms_per_token": round(1000 / tps, 3) if tps > 0 else float('inf'),
            "power_per_token_uj": round(vc.power_w * 1000000 / tps, 1) if tps > 0 else 0,
        }

    def compare_vessels(self) -> List[Dict]:
        return [self.throughput(v) for v in VESSEL_CONFIGS]


def demo():
    print("=== Neural Network Layer Simulator ===\n")

    # Quantization analysis
    print("--- Quantization Error Analysis ---")
    test_values = [random.gauss(0, 1) for _ in range(100)]
    for bits in [2, 4, 8, 16]:
        err = Quantizer.quantize_error(test_values, bits)
        print(f"  INT{bits}: MSE={err['mse']:.6f}, SNR={err['snr_db']:.1f}dB, MaxErr={err['max_error']:.4f}")
    print()

    # Single attention head
    print("--- Attention Head ---")
    head = AttentionHead(64)
    result = head.forward()
    print(f"  Score: {result['score']}")
    print(f"  Weights: {result['weights']}")
    print(f"  Output norm: {result['output_norm']}")
    print()

    # Model across vessels
    print("--- Model Throughput Comparison ---")
    for params_b in [1, 3, 7, 13]:
        d = max(256, int(params_b * 1000 ** 0.5))
        d_ff = d * 4
        heads = max(4, d // 64)
        layers = max(1, int(params_b * 1e9 // (12 * d * d)))
        model = ModelSimulator(f"{params_b}B", layers, d, d_ff, heads)
        print(f"\n  {model.name} ({model.total_params/1e9:.1f}B params, {layers} layers):")
        for row in model.compare_vessels():
            print(f"    {row['vessel']:10s}: {row['tokens_per_sec']:6.1f} tok/s, "
                  f"{row['latency_ms_per_token']:7.2f} ms, {row['power_per_token_uj']:8.1f} uJ/tok")


if __name__ == "__main__":
    demo()
