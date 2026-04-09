#!/usr/bin/env python3
"""Bitstream analyzer for mask-locked chip weight images.

Inspects compiled METL binary format, validates headers,
extracts layer metadata, and computes checksums.
"""
import struct, hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# METL format constants
METL_MAGIC = b"METL"
METL_VERSION = 1
METL_HEADER_SIZE = 64

# Precision codes
PREC_MAP = {0: "INT4", 1: "INT8", 2: "FP16", 3: "FP32", 4: "MIXED"}
PREC_BYTES = {"INT4": 0.5, "INT8": 1, "FP16": 2, "FP32": 4}


@dataclass
class LayerHeader:
    name: str
    precision: str
    rows: int
    cols: int
    offset: int
    size_bytes: int
    checksum: int


@dataclass
class BitstreamHeader:
    magic: bytes
    version: int
    num_layers: int
    model_name: str
    total_size: int
    creation_timestamp: int
    checksum: int


class BitstreamAnalyzer:
    """Analyze METL bitstream format."""

    def __init__(self):
        self.header: Optional[BitstreamHeader] = None
        self.layers: List[LayerHeader] = []
        self.raw_data: bytes = b""

    def generate_bitstream(self, model_name: str,
                            layers: List[Dict]) -> bytes:
        """Generate a valid METL bitstream for testing."""
        data = bytearray()

        # Header (64 bytes)
        header = struct.pack(">4sBI", METL_MAGIC, METL_VERSION, len(layers))
        name_bytes = model_name.encode()[:48].ljust(48, b'\x00')
        header += name_bytes
        header += struct.pack(">I", 0)  # timestamp placeholder
        header += struct.pack(">I", 0)  # checksum placeholder
        data.extend(header)

        # Layer headers (32 bytes each)
        layer_offset = METL_HEADER_SIZE + len(layers) * 32
        for layer in layers:
            prec_code = {v: k for k, v in PREC_MAP.items()}.get(
                layer.get("precision", "INT4"), 0)
            layer_data_size = int(layer["rows"] * layer["cols"] * PREC_BYTES.get(
                layer.get("precision", "INT4"), 1))

            lh = struct.pack(">16sBII",
                            layer["name"].encode()[:16].ljust(16, b'\x00'),
                            prec_code, layer["rows"], layer["cols"])
            lh += struct.pack(">II", layer_offset, layer_data_size)
            data.extend(lh)
            layer_offset += int(layer_data_size)

        # Layer data (zeros for test)
        for layer in layers:
            layer_data_size = int(layer["rows"] * layer["cols"] * PREC_BYTES.get(
                layer.get("precision", "INT4"), 1))
            data.extend(bytes(layer_data_size))

        # Fix total size
        struct.pack_into(">I", data, 52, len(data))

        # Fix header checksum
        h_hash = hashlib.sha256(data[:60]).digest()
        struct.pack_into(">I", data, 60, int.from_bytes(h_hash[:4], "big"))

        return bytes(data)

    def parse(self, data: bytes) -> bool:
        """Parse bitstream. Returns True if valid."""
        self.raw_data = data

        if len(data) < METL_HEADER_SIZE:
            return False

        # Header
        magic = data[0:4]
        if magic != METL_MAGIC:
            return False

        version = struct.unpack(">B", data[4:5])[0]
        if version != METL_VERSION:
            return False

        num_layers = struct.unpack(">I", data[5:9])[0]
        model_name = data[9:57].split(b"\\x00")[0].decode()
        total_size = struct.unpack(">I", data[52:56])[0]
        checksum = struct.unpack(">I", data[60:64])[0]

        self.header = BitstreamHeader(magic, version, num_layers,
                                      model_name, total_size, 0, checksum)

        # Layer headers
        self.layers = []
        offset = METL_HEADER_SIZE
        for i in range(num_layers):
            if offset + 32 > len(data):
                break
            name = data[offset:offset + 16].split(b"\\x00")[0].decode()
            prec_code = struct.unpack(">B", data[offset + 16:offset + 17])[0]
            rows, cols = struct.unpack(">II", data[offset + 17:offset + 25])
            layer_offset, layer_size = struct.unpack(">II", data[offset + 25:offset + 33])

            self.layers.append(LayerHeader(
                name=name,
                precision=PREC_MAP.get(prec_code, "UNKNOWN"),
                rows=rows, cols=cols,
                offset=layer_offset,
                size_bytes=layer_size,
                checksum=0
            ))
            offset += 32

        return True

    def validate_checksum(self) -> bool:
        if not self.raw_data:
            return False
        h_hash = hashlib.sha256(self.raw_data[:60]).digest()
        expected = int.from_bytes(h_hash[:4], "big")
        return expected == self.header.checksum

    def compute_layer_checksums(self) -> Dict[str, str]:
        """SHA256 for each layer's data."""
        result = {}
        for layer in self.layers:
            end = layer.offset + layer.size_bytes
            if end <= len(self.raw_data):
                h = hashlib.sha256(self.raw_data[layer.offset:end]).hexdigest()[:16]
                result[layer.name] = h
        return result

    def layer_sizes(self) -> List[Dict]:
        return [{"name": l.name, "precision": l.precision,
                 "shape": f"{l.rows}x{l.cols}",
                 "bytes": l.size_bytes} for l in self.layers]

    def total_weight_bytes(self) -> int:
        return sum(l.size_bytes for l in self.layers)

    def compression_ratio(self, original_params: int) -> float:
        return original_params * 4 / max(1, self.total_weight_bytes())


def demo():
    print("=== Bitstream Analyzer ===\n")

    # Generate test bitstream
    layers = [
        {"name": "attn_qkv", "rows": 512, "cols": 1536, "precision": "INT4"},
        {"name": "attn_out", "rows": 512, "cols": 512, "precision": "INT4"},
        {"name": "ffn_gate", "rows": 512, "cols": 2048, "precision": "INT4"},
        {"name": "ffn_up", "rows": 512, "cols": 2048, "precision": "INT8"},
        {"name": "ffn_down", "rows": 2048, "cols": 512, "precision": "INT4"},
        {"name": "layer_norm", "rows": 2, "cols": 512, "precision": "FP32"},
        {"name": "embed", "rows": 32000, "cols": 512, "precision": "INT8"},
    ]

    bs_data = BitstreamAnalyzer().generate_bitstream("frozen_scout_1b", layers)

    analyzer = BitstreamAnalyzer()
    valid = analyzer.parse(bs_data)
    print(f"Valid bitstream: {valid}")
    print(f"Checksum valid: {analyzer.validate_checksum()}")
    print()

    h = analyzer.header
    print(f"--- Header ---")
    print(f"  Magic: {h.magic}")
    print(f"  Version: {h.version}")
    print(f"  Model: {h.model_name}")
    print(f"  Layers: {h.num_layers}")
    print(f"  Total size: {h.total_size} bytes")
    print()

    print(f"--- Layers ---")
    total = 0
    for l in analyzer.layer_sizes():
        print(f"  {l['name']:15s}: {l['shape']:>10s} {l['precision']:>5s} = {l['bytes']:>8d} bytes")
        total += l["bytes"]
    print(f"  {'TOTAL':15s}: {'':>10s} {'':>5s} = {total:>8d} bytes")
    print(f"  Weight memory: {total / 1024:.1f} KB")

    orig_params = sum(l["rows"] * l["cols"] for l in layers)
    ratio = analyzer.compression_ratio(orig_params)
    print(f"  Compression: {ratio:.1f}x vs FP32 ({orig_params * 4 / 1024:.1f} KB -> {total / 1024:.1f} KB)")
    print()

    # Layer checksums
    print("--- Layer Checksums ---")
    checksums = analyzer.compute_layer_checksums()
    for name, cksum in checksums.items():
        print(f"  {name:15s}: {cksum}")


if __name__ == "__main__":
    demo()
