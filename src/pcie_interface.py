#!/usr/bin/env python3
"""PCIe interface simulator for mask-locked inference chips.

Simulates the host-to-chip PCIe bus: config space, BAR mapping,
DMA transfers, and interrupt handling. Zero external dependencies.
"""
import struct, time, random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import IntEnum, Enum


class PCICommand(IntEnum):
    IO_SPACE = 0x0001
    MEMORY_SPACE = 0x0002
    BUS_MASTER = 0x0004
    SPECIAL_CYCLES = 0x0008
    MEM_WRITE_INVALIDATE = 0x0010
    VGA_PALETTE_SNOOP = 0x0020
    PARITY_ERROR = 0x0040
    WAIT_CYCLE = 0x0080
    SERR = 0x0100
    FAST_BACK_TO_BACK = 0x0200


class PCIStatus(IntEnum):
    CAPABILITIES = 0x0010
    FREQ_66MHZ = 0x0020
    FAST_BACK = 0x0080
    DEVSEL = 0x0600  # bits 8-9


class BARType(Enum):
    MEMORY_32 = "mem32"
    MEMORY_64 = "mem64"
    IO = "io"


@dataclass
class BAR:
    index: int
    bar_type: BARType
    size: int  # bytes
    base_addr: int = 0
    prefetchable: bool = False

    @property
    def mask(self) -> int:
        """BAR size mask for sizing."""
        size = self.size
        # Round up to power of 2
        p = 1
        while p < size:
            p *= 2
        return ~(p - 1) & 0xFFFFFFFF


@dataclass
class DMATransfer:
    src_addr: int
    dst_addr: int
    size: int
    direction: str  # "h2d" (host to device) or "d2h" (device to host)
    completed: bool = False
    error: bool = False
    bytes_transferred: int = 0


class PCIeConfigSpace:
    """PCIe configuration space (256 bytes)."""

    def __init__(self, vendor_id: int = 0x1AF4, device_id: int = 0x1337,
                 class_code: int = 0x068000,  # Bridge/Other
                 revision: int = 1):
        self.data = bytearray(256)
        # Vendor/Device ID
        struct.pack_into("<HH", self.data, 0x00, vendor_id, device_id)
        # Command/Status
        struct.pack_into("<HH", self.data, 0x04, PCICommand.MEMORY_SPACE | PCICommand.BUS_MASTER, 0)
        # Revision/Class
        struct.pack_into("<HI", self.data, 0x08, revision, class_code)
        # Cache Line Size
        struct.pack_into("<H", self.data, 0x0C, 64)
        # BARs (initialized to 0)
        # Subsystem
        struct.pack_into("<HH", self.data, 0x2C, vendor_id, device_id)
        # Expansion ROM base
        struct.pack_into("<I", self.data, 0x30, 0)
        # Capabilities pointer
        self.data[0x34] = 0x40
        # MSI capability at 0x40
        self._init_msi()

    def _init_msi(self):
        """Initialize MSI capability."""
        self.data[0x40] = 0x05  # MSI capability ID
        self.data[0x41] = 0x00  # Next capability pointer (none)
        self.data[0x42] = 0x01  # MSI enable, 64-bit capable

    def read(self, offset: int, size: int = 4) -> int:
        if offset + size > 256:
            return 0
        return int.from_bytes(self.data[offset:offset + size], "little")

    def write(self, offset: int, value: int, size: int = 4):
        if offset + size > 256:
            return
        data = value.to_bytes(size, "little")
        self.data[offset:offset + size] = data


class PCIeDevice:
    """Simulated PCIe device for mask-locked inference chip."""

    def __init__(self, name: str = "frozen_intelligence"):
        self.name = name
        self.config = PCIeConfigSpace()
        self.bars: Dict[int, BAR] = {}
        self.memory: Dict[int, bytearray] = {}  # BAR index -> memory
        self.dma_queue: List[DMATransfer] = []
        self.dma_callback: Optional[Callable] = None
        self.interrupt_pending = False
        self._setup_bars()

    def _setup_bars(self):
        """Set up default BARs."""
        # BAR0: Inference registers (256KB)
        bar0 = BAR(0, BARType.MEMORY_32, 256 * 1024)
        self.bars[0] = bar0
        self.memory[0] = bytearray(256 * 1024)

        # BAR1: Weight memory (4MB)
        bar1 = BAR(1, BARType.MEMORY_32, 4 * 1024 * 1024)
        self.bars[1] = bar1
        self.memory[1] = bytearray(4 * 1024 * 1024)

        # BAR2: KV cache (8MB)
        bar2 = BAR(2, BARType.MEMORY_32, 8 * 1024 * 1024)
        self.bars[2] = bar2
        self.memory[2] = bytearray(8 * 1024 * 1024)

        # BAR3: Status/config (4KB)
        bar3 = BAR(3, BARType.MEMORY_32, 4 * 1024)
        self.bars[3] = bar3
        self.memory[3] = bytearray(4 * 1024)

        # Initialize BAR values in config space
        for bar in self.bars.values():
            self.config.write(0x10 + bar.index * 4, bar.base_addr)

    # Register offsets in BAR0 (inference registers)
    REG_CONTROL = 0x00
    REG_STATUS = 0x04
    REG_INPUT_ADDR = 0x08
    REG_OUTPUT_ADDR = 0x0C
    REG_TOKEN_COUNT = 0x10
    REG_TEMPERATURE = 0x14
    REG_POWER = 0x18
    REG_CLOCK = 0x1C
    REG_LAYER = 0x20
    REG_CHIP_ID = 0x24

    # Control bits
    CTRL_START = 0x01
    CTRL_RESET = 0x02
    CTRL_STREAM = 0x04

    # Status bits
    STATUS_BUSY = 0x01
    STATUS_DONE = 0x02
    STATUS_ERROR = 0x04
    STATUS_THERMAL = 0x08

    def mmio_read(self, bar_idx: int, offset: int, size: int = 4) -> int:
        """Memory-mapped IO read."""
        if bar_idx not in self.memory:
            return 0
        mem = self.memory[bar_idx]
        if offset + size > len(mem):
            return 0
        return int.from_bytes(mem[offset:offset + size], "little")

    def mmio_write(self, bar_idx: int, offset: int, value: int, size: int = 4):
        """Memory-mapped IO write."""
        if bar_idx not in self.memory:
            return
        mem = self.memory[bar_idx]
        if offset + size > len(mem):
            return
        data = value.to_bytes(size, "little")
        mem[offset:offset + size] = data

        # Handle special register writes
        if bar_idx == 0:
            self._handle_register_write(offset, value)

    def _handle_register_write(self, offset: int, value: int):
        """Handle writes to control registers."""
        if offset == self.REG_CONTROL:
            if value & self.CTRL_RESET:
                self._do_reset()
            elif value & self.CTRL_START:
                self._do_inference(value & self.CTRL_STREAM)

    def _do_reset(self):
        """Reset chip state."""
        for bar_mem in self.memory.values():
            for i in range(0, min(256, len(bar_mem)), 4):
                bar_mem[i:i+4] = struct.pack("<I", 0)
        self.interrupt_pending = False

    def _do_inference(self, streaming: bool = False):
        """Simulate inference (mock)."""
        # Set busy
        status = self.mmio_read(0, self.REG_STATUS)
        self.mmio_write(0, self.REG_STATUS, status | self.STATUS_BUSY)

        # Simulate work (update token count)
        tokens = self.mmio_read(0, self.REG_TOKEN_COUNT)
        self.mmio_write(0, self.REG_TOKEN_COUNT, tokens + 1)

        # Set done
        status = self.mmio_read(0, self.REG_STATUS)
        self.mmio_write(0, self.REG_STATUS, (status & ~self.STATUS_BUSY) | self.STATUS_DONE)
        self.interrupt_pending = True

    def queue_dma(self, transfer: DMATransfer) -> int:
        """Queue a DMA transfer. Returns transfer index."""
        idx = len(self.dma_queue)
        self.dma_queue.append(transfer)
        return idx

    def process_dma(self, transfer_idx: int) -> bool:
        """Process a DMA transfer (simulated)."""
        if transfer_idx >= len(self.dma_queue):
            return False
        t = self.dma_queue[transfer_idx]
        # Simulate DMA latency
        t.bytes_transferred = t.size
        t.completed = True
        if self.dma_callback:
            self.dma_callback(t)
        return True

    def read_chip_info(self) -> Dict:
        """Read chip identification."""
        chip_id = self.mmio_read(0, self.REG_CHIP_ID)
        clock = self.mmio_read(0, self.REG_CLOCK)
        temp = self.mmio_read(0, self.REG_TEMPERATURE)
        power = self.mmio_read(0, self.REG_POWER)
        return {"chip_id": chip_id, "clock_mhz": clock, "temperature_c": temp, "power_mw": power}


class PCIeHost:
    """Host-side PCIe driver."""

    def __init__(self, device: PCIeDevice):
        self.device = device
        self.base_addr = 0xFF000000  # Host-mapped base address

    def config_read(self, offset: int, size: int = 4) -> int:
        return self.device.config.read(offset, size)

    def config_write(self, offset: int, value: int, size: int = 4):
        self.device.config.write(offset, value, size)

    def read32(self, bar: int, offset: int) -> int:
        return self.device.mmio_read(bar, offset, 4)

    def write32(self, bar: int, offset: int, value: int):
        self.device.mmio_write(bar, offset, value, 4)

    def read_buffer(self, bar: int, offset: int, size: int) -> bytes:
        """Read a buffer from device memory."""
        data = bytearray(size)
        for i in range(0, size, 4):
            val = self.device.mmio_read(bar, offset + i, 4)
            chunk = min(4, size - i)
            data[i:i+chunk] = val.to_bytes(4, "little")[:chunk]
        return bytes(data)

    def write_buffer(self, bar: int, offset: int, data: bytes):
        """Write a buffer to device memory."""
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if len(chunk) < 4:
                chunk = chunk + b"\\x00" * (4 - len(chunk))
            val = int.from_bytes(chunk, "little")
            self.device.mmio_write(bar, offset + i, val, 4)

    def start_inference(self, input_addr: int, output_addr: int,
                        tokens: int, streaming: bool = False):
        """Start inference on the device."""
        ctrl = PCIeDevice.CTRL_START | (PCIeDevice.CTRL_STREAM if streaming else 0)
        self.write32(0, PCIeDevice.REG_INPUT_ADDR, input_addr)
        self.write32(0, PCIeDevice.REG_OUTPUT_ADDR, output_addr)
        self.write32(0, PCIeDevice.REG_TOKEN_COUNT, tokens)
        self.write32(0, PCIeDevice.REG_CONTROL, ctrl)

    def wait_complete(self, timeout_ms: int = 1000) -> bool:
        """Wait for inference to complete."""
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            status = self.read32(0, PCIeDevice.REG_STATUS)
            if status & PCIeDevice.STATUS_DONE:
                return True
            if status & PCIeDevice.STATUS_ERROR:
                return False
            time.sleep(0.001)  # 1ms poll
        return False

    def dma_transfer(self, bar: int, offset: int, data: bytes,
                     direction: str = "h2d") -> bool:
        """Transfer data via DMA."""
        t = DMATransfer(src_addr=0, dst_addr=offset, size=len(data), direction=direction)
        idx = self.device.queue_dma(t)
        if direction == "h2d":
            self.write_buffer(bar, offset, data)
        else:
            data = self.read_buffer(bar, offset, len(data))
        return self.device.process_dma(idx)


def demo():
    print("=== PCIe Interface Simulator ===\n")

    # Create device and host
    device = PCIeDevice("frozen_intelligence_scout")
    host = PCIeHost(device)

    # Config space
    print("--- Config Space ---")
    vendor = host.config_read(0x00, 2)
    device_id = host.config_read(0x02, 2)
    class_code = host.config_read(0x09, 3)
    print(f"  Vendor: 0x{vendor:04X}, Device: 0x{device_id:04X}")
    print(f"  Class: 0x{class_code:06X}")
    print(f"  Command: 0x{host.config_read(0x04, 2):04X}")
    print()

    # BARs
    print("--- BARs ---")
    for idx, bar in device.bars.items():
        print(f"  BAR{idx}: {bar.bar_type.value}, {bar.size/1024:.0f}KB @ 0x{bar.base_addr:08X}")
    print()

    # MMIO operations
    print("--- MMIO Operations ---")
    # Write chip ID
    host.write32(0, device.REG_CHIP_ID, 0xF1000001)
    host.write32(0, device.REG_CLOCK, 500)
    host.write32(0, device.REG_TEMPERATURE, 45)

    info = device.read_chip_info()
    print(f"  Chip ID: 0x{info['chip_id']:08X}")
    print(f"  Clock: {info['clock_mhz']} MHz")
    print(f"  Temperature: {info['temperature_c']}°C")
    print()

    # Inference
    print("--- Inference ---")
    host.start_inference(0x1000, 0x2000, 64)
    complete = host.wait_complete(100)
    status = host.read32(0, device.REG_STATUS)
    tokens = host.read32(0, device.REG_TOKEN_COUNT)
    print(f"  Started: control=0x{PCIeDevice.CTRL_START:02X}")
    print(f"  Complete: {complete}")
    print(f"  Status: 0x{status:04X}, Tokens: {tokens}")
    print()

    # DMA transfer
    print("--- DMA Transfer ---")
    test_data = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    success = host.dma_transfer(1, 0, test_data, "h2d")
    print(f"  Write {len(test_data)} bytes: {'OK' if success else 'FAIL'}")

    readback = host.read_buffer(1, 0, 8)
    print(f"  Readback: {readback.hex()}")
    print(f"  Match: {readback == test_data}")
    print()

    # Bandwidth estimate
    print("--- Bandwidth Estimate ---")
    for size_kb in [64, 256, 1024]:
        size = size_kb * 1024
        data = bytes(size)
        start = time.time()
        host.dma_transfer(1, 0, data, "h2d")
        elapsed = time.time() - start
        bw = size / elapsed / 1e9
        print(f"  {size_kb:4d}KB: {elapsed*1000:.1f}ms, {bw:.2f} GB/s")


if __name__ == "__main__":
    demo()
