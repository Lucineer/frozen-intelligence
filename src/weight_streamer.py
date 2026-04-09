#!/usr/bin/env python3
"""Weight streaming controller — DDR4 → BRAM for FPGA prototypes.

Based on SuperInstance FPGA Prototype Implementation Guide.
"""
import struct, time
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class DDR4Interface:
    """Simulated DDR4 memory interface."""

    def __init__(self, capacity_mb: int = 1024, bandwidth_gbs: float = 19.2):
        self.capacity = capacity_mb * 1024 * 1024
        self.bandwidth = bandwidth_gbs * 1e9 / 8  # bytes/sec
        self.data = bytearray(self.capacity)
        self.latency_ns = 50  # CAS latency

    def read(self, addr: int, size: int) -> bytes:
        """Read from DDR4 with simulated latency."""
        time.sleep(self.latency_ns / 1e9)
        if addr + size > len(self.data):
            return b"\x00" * size
        return bytes(self.data[addr:addr + size])

    def write(self, addr: int, data: bytes):
        """Write to DDR4."""
        if addr + len(data) <= len(self.data):
            self.data[addr:addr + len(data)] = data

    def load_weights(self, weights: bytes, base_addr: int = 0):
        """Load weight binary into DDR4."""
        self.write(base_addr, weights)


class BRAMBank:
    """Simulated BRAM bank (36Kb)."""

    def __init__(self, depth: int = 1024, width_bytes: int = 4):
        self.depth = depth
        self.width = width_bytes
        self.data = [0] * (depth * width_bytes)
        self.read_latency_cycles = 2

    def write(self, addr: int, data: bytes):
        """Write to BRAM."""
        offset = addr * self.width
        for i, byte in enumerate(data):
            if offset + i < len(self.data):
                self.data[offset + i] = byte

    def read(self, addr: int, words: int = 1) -> bytes:
        """Read from BRAM."""
        offset = addr * self.width
        end = offset + words * self.width
        return bytes(self.data[offset:end])


@dataclass
class LayerDescriptor:
    name: str
    base_addr_ddr: int  # DDR4 address
    size_bytes: int
    bram_bank: int
    bram_offset: int
    rows: int
    cols: int
    precision_bits: int = 4


class WeightStreamer:
    """Controller for streaming weights from DDR4 to BRAM."""

    def __init__(self, ddr: DDR4Interface, bram_banks: List[BRAMBank],
                 bus_width_bytes: int = 16):
        self.ddr = ddr
        self.brams = bram_banks
        self.bus_width = bus_width_bytes
        self.state = "IDLE"
        self.current_layer: Optional[LayerDescriptor] = None
        self.bytes_transferred = 0
        self.start_time = 0

    def start_load(self, layer: LayerDescriptor) -> bool:
        """Start loading a layer's weights."""
        if self.state != "IDLE":
            return False
        self.current_layer = layer
        self.state = "REQUEST"
        self.bytes_transferred = 0
        self.start_time = time.time()
        return True

    def step(self) -> bool:
        """Execute one streaming step. Returns True if still loading."""
        if self.state == "IDLE" or not self.current_layer:
            return False

        layer = self.current_layer
        if self.state == "REQUEST":
            # Read from DDR4
            addr = layer.base_addr_ddr + self.bytes_transferred
            read_size = min(self.bus_width, layer.size_bytes - self.bytes_transferred)
            if read_size <= 0:
                self.state = "DONE"
                return False
            self.ddr_data = self.ddr.read(addr, read_size)
            self.state = "WRITE_BRAM"
            return True

        elif self.state == "WRITE_BRAM":
            # Write to BRAM
            bram = self.brams[layer.bram_bank]
            bram_addr = layer.bram_offset + self.bytes_transferred // self.bus_width
            bram.write(bram_addr, self.ddr_data)
            self.bytes_transferred += len(self.ddr_data)
            if self.bytes_transferred >= layer.size_bytes:
                self.state = "DONE"
                return False
            self.state = "REQUEST"
            return True

        return False

    def load_complete(self) -> bool:
        return self.state == "DONE"

    def bandwidth_utilization(self) -> float:
        """Calculate actual vs theoretical bandwidth."""
        if not self.start_time or self.bytes_transferred == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        actual_bw = self.bytes_transferred / elapsed
        return actual_bw / (self.ddr.bandwidth)

    def estimate_load_time(self, layer: LayerDescriptor) -> float:
        """Estimate time to load layer (seconds)."""
        # DDR4 bandwidth minus overhead
        effective_bw = self.ddr.bandwidth * 0.7  # 70% efficiency
        return layer.size_bytes / effective_bw


class MultiLayerStreamer:
    """Orchestrates streaming for entire model."""

    def __init__(self, ddr_capacity_mb: int = 1024):
        self.ddr = DDR4Interface(ddr_capacity_mb)
        # 8 BRAM banks (typical for KV260)
        self.brams = [BRAMBank(depth=1024, width_bytes=4) for _ in range(8)]
        self.streamer = WeightStreamer(self.ddr, self.brams)
        self.layers: List[LayerDescriptor] = []
        self.model_weights: bytes = b""

    def add_layer(self, name: str, weights: bytes, rows: int, cols: int,
                  precision_bits: int = 4) -> int:
        """Add layer to model. Returns layer index."""
        layer_idx = len(self.layers)
        base_addr = len(self.model_weights)
        size = len(weights)
        # Assign BRAM bank round-robin
        bram_bank = layer_idx % len(self.brams)
        bram_offset = 0  # Simplified
        layer = LayerDescriptor(name=name, base_addr_ddr=base_addr,
                               size_bytes=size, bram_bank=bram_bank,
                               bram_offset=bram_offset, rows=rows, cols=cols,
                               precision_bits=precision_bits)
        self.layers.append(layer)
        self.model_weights += weights
        return layer_idx

    def load_model(self):
        """Load entire model into DDR4."""
        self.ddr.load_weights(self.model_weights)

    def stream_layer(self, layer_idx: int) -> Dict:
        """Stream a single layer, return metrics."""
        layer = self.layers[layer_idx]
        self.streamer.start_load(layer)
        steps = 0
        while self.streamer.step():
            steps += 1
        elapsed = time.time() - self.streamer.start_time
        bw_util = self.streamer.bandwidth_utilization()
        return {"layer": layer.name, "size_mb": layer.size_bytes / 1e6,
                "steps": steps, "time_s": round(elapsed, 3),
                "bw_util": round(bw_util * 100, 1)}

    def pipeline_stream(self, active_layers: List[int]) -> Dict:
        """Stream multiple layers in pipeline."""
        # Simplified pipeline simulation
        total_size = sum(self.layers[i].size_bytes for i in active_layers)
        # Pipeline efficiency: 80% overlap
        serial_time = sum(self.streamer.estimate_load_time(self.layers[i])
                         for i in active_layers)
        pipelined_time = serial_time * 0.8
        return {"layers": len(active_layers), "total_mb": total_size / 1e6,
                "serial_s": round(serial_time, 3),
                "pipelined_s": round(pipelined_time, 3),
                "speedup": round(serial_time / pipelined_time, 2)}


def demo():
    print("=== Weight Streaming Controller (DDR4 → BRAM) ===\n")

    # Create streamer
    streamer = MultiLayerStreamer(ddr_capacity_mb=1024)
    print(f"DDR4: 1GB, {streamer.ddr.bandwidth/1e9:.1f} GB/s theoretical")
    print(f"BRAM: {len(streamer.brams)} banks × 36Kb each")
    print()

    # Add layers (simulated weights)
    layer_sizes = [
        ("attn_qkv", 1.5, 2048, 2048),
        ("attn_out", 0.5, 2048, 2048),
        ("ffn_up", 2.0, 2048, 8192),
        ("ffn_down", 2.0, 8192, 2048),
    ]
    for name, mb, rows, cols in layer_sizes:
        size = int(mb * 1024 * 1024)
        weights = bytes([i % 256 for i in range(size)])  # Simulated
        idx = streamer.add_layer(name, weights, rows, cols, 4)
        print(f"  Layer {idx}: {name:12s} {rows:4d}×{cols:<4d} {mb:5.1f}MB")

    # Load model into DDR4
    streamer.load_model()
    print(f"\nTotal model: {len(streamer.model_weights)/1e6:.1f} MB in DDR4")
    print()

    # Stream individual layer
    print("--- Streaming Layer 0 (attn_qkv) ---")
    metrics = streamer.stream_layer(0)
    print(f"  Size: {metrics['size_mb']:.1f} MB")
    print(f"  Time: {metrics['time_s']:.3f}s")
    print(f"  BW utilization: {metrics['bw_util']}%")
    print(f"  Steps: {metrics['steps']}")
    print()

    # Pipeline streaming
    print("--- Pipeline Streaming (layers 0-3) ---")
    pipe = streamer.pipeline_stream([0, 1, 2, 3])
    print(f"  {pipe['layers']} layers, {pipe['total_mb']:.1f} MB total")
    print(f"  Serial: {pipe['serial_s']:.3f}s")
    print(f"  Pipelined: {pipe['pipelined_s']:.3f}s")
    print(f"  Speedup: {pipe['speedup']:.1f}x")
    print()

    # KV260 resource check
    print("--- KV260 Resource Fit ---")
    kv260_bram = 135  # BRAM36K blocks
    kv260_luts = 117000
    # TLMM array for 64×64
    from tlmm_engine import TLMMArray, TLMMConfig
    config = TLMMConfig()
    array = TLMMArray(64, 64, config)
    res = array.resource_estimate()
    print(f"  64×64 TLMM array: {res['luts']:,} LUTs, {res['bram36k']} BRAM36K")
    print(f"  KV260 has: {kv260_luts:,} LUTs, {kv260_bram} BRAM36K")
    print(f"  Fit: {'YES' if res['luts'] < kv260_luts and res['bram36k'] < kv260_bram else 'NO'}")
    print()

    # Performance estimate
    print("--- Performance Estimate ---")
    # 64×64 array @ 500MHz, ternary weights
    ops_per_cycle = 64 * 64 * 2  # 2 ops per MAC (multiply + add)
    gops = ops_per_cycle * 500e6 / 1e9
    print(f"  64×64 TLMM array @ 500MHz: {gops:.1f} GOPS")
    print(f"  Power estimate: ~{res['pe_count'] * 0.001:.1f}W")
    print(f"  Throughput: ~{gops / 13:.1f} tokens/s for 13B model")


if __name__ == "__main__":
    demo()
