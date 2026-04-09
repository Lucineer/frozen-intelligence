#!/usr/bin/env python3
"""Weight-to-Metal compiler — converts trained model weights to quantized metal encoding."""
import struct, math, json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BINARY = "binary"


@dataclass
class QuantizedWeight:
    """A single quantized weight value with metadata."""
    raw: float
    quantized: int
    bits: int
    scale: float
    zero_point: float
    layer_name: str
    index: Tuple[int, ...]

    @property
    def bits_per_weight(self) -> int:
        return self.bits

    def to_metal_encoding(self) -> bytes:
        """Encode as the binary pattern that would become metal interconnect."""
        if self.bits == 4:
            return struct.pack("B", self.quantized & 0x0F)
        elif self.bits == 8:
            return struct.pack("b", self.quantized)
        elif self.bits == 2:
            return struct.pack("B", self.quantized & 0x03)
        elif self.bits == 1:
            return struct.pack("B", 1 if self.quantized > 0 else 0)
        return struct.pack("f", self.raw)


@dataclass
class LayerSpec:
    """Specification for a single neural network layer."""
    name: str
    shape: Tuple[int, ...]
    precision: Precision = Precision.INT4
    scale: float = 1.0
    zero_point: float = 0.0
    mixed_precision: Optional[Dict[str, Precision]] = None

    @property
    def num_weights(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result

    @property
    def bits_total(self) -> int:
        return self.num_weights * self.precision.value.count("int") or self.num_weights * 4

    @property
    def bytes_total(self) -> int:
        return math.ceil(self.bits_total / 8)

    def estimate_die_area_mm2(self, process_nm: int = 28) -> float:
        """Rough die area estimate in mm^2 for this layer."""
        bits = self.num_weights * {"int4": 4, "int2": 2, "int8": 8, "binary": 1,
                                    "fp16": 16, "fp32": 32}.get(
            self.precision.value, 4)
        gate_pitch_nm = process_nm * 2.5
        area_nm2 = bits * (gate_pitch_nm ** 2) * 0.15
        return area_nm2 * 1e-12 * 1e6  # nm^2 to mm^2


class WeightQuantizer:
    """Quantizes floating-point weights to target precision."""

    def __init__(self, precision: Precision = Precision.INT4):
        self.precision = precision

    def compute_scale(self, weights: List[float]) -> Tuple[float, float]:
        wmin, wmax = min(weights), max(weights)
        if self.precision == Precision.INT4:
            qmin, qmax = -8, 7
        elif self.precision == Precision.INT2:
            qmin, qmax = -2, 1
        elif self.precision == Precision.INT8:
            qmin, qmax = -128, 127
        elif self.precision == Precision.BINARY:
            return 1.0, 0.0
        else:
            qmin, qmax = 0, 255
        scale = (wmax - wmin) / (qmax - qmin) if qmax != qmin else 1.0
        zero_point = qmin - wmin / scale
        return scale, zero_point

    def quantize(self, weights: List[float], layer_name: str = "",
                 shape: Optional[Tuple[int, ...]] = None) -> List[QuantizedWeight]:
        scale, zp = self.compute_scale(weights)
        if self.precision == Precision.BINARY:
            return [QuantizedWeight(
                raw=w, quantized=1 if w > 0 else -1, bits=1,
                scale=1.0, zero_point=0.0, layer_name=layer_name,
                index=self._idx(i, shape)) for i, w in enumerate(weights)]

        if self.precision == Precision.INT4:
            qmin, qmax = -8, 7
        elif self.precision == Precision.INT2:
            qmin, qmax = -2, 1
        elif self.precision == Precision.INT8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 255

        result = []
        for i, w in enumerate(weights):
            q = int(round(w / scale + zp))
            q = max(qmin, min(qmax, q))
            result.append(QuantizedWeight(
                raw=w, quantized=q, bits={"INT4": 4, "INT2": 2, "INT8": 8}.get(
                    self.precision.name, 4),
                scale=scale, zero_point=zp, layer_name=layer_name,
                index=self._idx(i, shape)))
        return result

    def _idx(self, flat: int, shape=None) -> Tuple[int, ...]:
        if shape is None:
            return (flat,)
        idx = []
        remaining = flat
        for dim in reversed(shape):
            idx.append(remaining % dim)
            remaining //= dim
        return tuple(reversed(idx))


class MetalEncoder:
    """Encodes quantized weights into the binary format for mask fabrication."""

    HEADER_MAGIC = b"METL"
    VERSION = 1

    def encode_layer(self, weights: List[QuantizedWeight]) -> bytes:
        """Encode a layer into metal binary format."""
        if not weights:
            return b""
        bits = weights[0].bits
        data = bytearray()

        for w in weights:
            data.extend(w.to_metal_encoding())

        return bytes(data)

    def encode_chip(self, layers: Dict[str, List[QuantizedWeight]],
                    metadata: Optional[Dict] = None) -> bytes:
        """Encode entire chip with header, layer table, and weight data."""
        output = bytearray()
        output.extend(self.HEADER_MAGIC)
        output.extend(struct.pack("<H", self.VERSION))
        output.extend(struct.pack("<H", len(layers)))

        layer_offsets = []
        data_offset = 8 + len(layers) * 28

        for name, weights in layers.items():
            bits = weights[0].bits if weights else 4
            offset = data_offset
            size = len(self.encode_layer(weights))
            safe_name = name[:12].encode("ascii", errors="replace").ljust(12, b"\x00")
            layer_offsets.append((safe_name, offset, size,
                                  len(weights), bits))
            data_offset += size

        for name_bytes, offset, size, count, bits in layer_offsets:
            output.extend(name_bytes)
            output.extend(struct.pack("<III", offset, size, count))
            output.extend(struct.pack("<B", bits))
            output.extend(b"\x00" * 3)

        for name, weights in layers.items():
            output.extend(self.encode_layer(weights))

        if metadata:
            meta_json = json.dumps(metadata).encode()
            output.extend(struct.pack("<I", len(meta_json)))
            output.extend(meta_json)

        return bytes(output)

    def decode_stats(self, chip_bytes: bytes) -> Dict:
        """Decode chip header for inspection."""
        if chip_bytes[:4] != self.HEADER_MAGIC:
            return {"error": "invalid magic"}
        version = struct.unpack("<H", chip_bytes[4:6])[0]
        num_layers = struct.unpack("<H", chip_bytes[6:8])[0]
        layers = []
        for i in range(num_layers):
            base = 8 + i * 28
            raw = chip_bytes[base:base+12].rstrip(b"\x00")
            name = raw.decode("ascii", errors="replace")
            offset, size, count = struct.unpack("<III", chip_bytes[base+12:base+24])
            bits = chip_bytes[base+24]
            layers.append({"name": name, "offset": offset, "size": size,
                           "count": count, "bits": bits})
        return {"version": version, "num_layers": num_layers, "layers": layers,
                "total_bytes": len(chip_bytes)}


class ChipEstimator:
    """Estimates physical chip parameters."""

    DENSITY_TABLE = {
        28: 91.0,   # million transistors per mm^2
        40: 51.0,
        65: 28.0,
        7: 180.0,
    }

    POWER_TABLE = {
        28: {"dynamic_mw_per_ghz_mm2": 0.5, "static_uw_per_mm2": 10},
        40: {"dynamic_mw_per_ghz_mm2": 0.3, "static_uw_per_mm2": 6},
        65: {"dynamic_mw_per_ghz_mm2": 0.15, "static_uw_per_mm2": 3},
    }

    def __init__(self, process_nm: int = 28):
        self.process = process_nm
        self.density = self.DENSITY_TABLE.get(process_nm, 28)  # mTr/mm^2

    def estimate_die_area(self, total_weights: int, bits_per_weight: int = 4) -> float:
        total_bits = total_weights * bits_per_weight
        memory_area = total_bits / (self.density * 1e6) * 3.0
        compute_area = memory_area * 0.4
        control_area = memory_area * 0.1
        io_area = 2.0
        total = memory_area + compute_area + control_area + io_area
        return round(total, 2)

    def estimate_power(self, die_area_mm2: float, clock_ghz: float = 0.5,
                       utilization: float = 0.7) -> Dict:
        power = self.POWER_TABLE.get(self.process, self.POWER_TABLE[28])
        dynamic = power["dynamic_mw_per_ghz_mm2"] * clock_ghz * die_area_mm2 * utilization
        static = power["static_uw_per_mm2"] * die_area_mm2 / 1000
        return {"dynamic_mw": round(dynamic, 1), "static_mw": round(static, 2),
                "total_mw": round(dynamic + static, 1), "watts": round((dynamic + static) / 1000, 2)}

    def estimate_throughput(self, model_params_b: float, die_area_mm2: float,
                            bits: int = 4) -> Dict:
        weight_memory_bw_tb_s = die_area_mm2 * self.density * 0.1 / 8e6
        ops_per_token = model_params_b * 2e9
        latency_s = ops_per_token / (weight_memory_bw_tb_s * 1e12 * 0.5)
        tok_per_s = 1.0 / latency_s if latency_s > 0 else 0
        return {"weight_bw_tb_s": round(weight_memory_bw_tb_s, 1),
                "tok_per_s": round(min(tok_per_s, 200), 1),
                "latency_ms_per_tok": round(latency_s * 1000, 2)}

    def full_report(self, model_params_b: float, bits: int = 4,
                    clock_ghz: float = 0.5) -> Dict:
        area = self.estimate_die_area(int(model_params_b * 1e9), bits)
        power = self.estimate_power(area, clock_ghz)
        throughput = self.estimate_throughput(model_params_b, area, bits)
        return {"model_params_b": model_params_b, "quantization": f"INT{bits}",
                "process_nm": self.process, "die_area_mm2": area,
                "die_size_mm": round(math.sqrt(area), 1), **power, **throughput}


def demo():
    print("=== Frozen Intelligence: Weight-to-Metal Compiler ===\n")

    # Simulated transformer weights (small demo)
    q = WeightQuantizer(Precision.INT4)
    enc = MetalEncoder()
    est = ChipEstimator(28)

    # Create demo layers
    import random
    random.seed(42)

    layers = {}
    layer_specs = [
        ("attention_qkv", (12, 256, 768)),
        ("attention_out", (12, 768, 768)),
        ("ffn_up", (768, 3072)),
        ("ffn_down", (3072, 768)),
        ("lm_head", (768, 32000)),
    ]

    print("Quantizing layers:")
    total_weights = 0
    for name, shape in layer_specs:
        n = 1
        for d in shape:
            n *= d
        weights = [random.gauss(0, 0.05) for _ in range(min(n, 1000))]  # demo subset
        quantized = q.quantize(weights, name, shape[:2])
        layers[name] = quantized
        total_weights += n
        print(f"  {name}: {shape} -> {n:,} weights ({min(n,1000)} sampled)")

    # Encode
    chip_bytes = enc.encode_chip(layers, {"model": "demo-3b", "process": "28nm"})
    stats = enc.decode_stats(chip_bytes)
    print(f"\nChip binary: {len(chip_bytes):,} bytes")
    print(f"Layers: {stats['num_layers']}")
    for layer in stats["layers"]:
        print(f"  {layer['name']}: {layer['count']} weights @ {layer['bits']}bit")

    # Die estimation for product line
    print("\n=== Product Line Estimates (28nm) ===")
    products = [
        ("Scout", 1.0, 2),
        ("Messenger", 3.0, 4),
        ("Navigator", 7.0, 4),
        ("Captain", 13.0, 4),
    ]
    for name, params, bits in products:
        r = est.full_report(params, bits)
        print(f"\n{name} ({params}B INT{bits}):")
        print(f"  Die: {r['die_area_mm2']}mm² ({r['die_size_mm']}mm × {r['die_size_mm']}mm)")
        print(f"  Power: {r['watts']}W")
        print(f"  Throughput: {r['tok_per_s']} tok/s")
        print(f"  Weight BW: {r['weight_bw_tb_s']} TB/s")


if __name__ == "__main__":
    demo()
