#!/usr/bin/env python3
"""Memory controller for mask-locked inference chips.

DDR4/LPDDR4 memory controller simulation: command scheduling,
refresh management, bank interleaving, and bandwidth optimization
for weight streaming.
"""
import math, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum, IntEnum


class DRAMCommand(IntEnum):
    ACTIVATE = 1
    READ = 2
    WRITE = 3
    PRECHARGE = 4
    REFRESH = 5
    ACT_READ = 6
    ACT_WRITE = 7


class DRAMTiming:
    """Timing parameters for DDR4/LPDDR4."""

    def __init__(self, freq_mhz: int = 1600, dram_type: str = "DDR4"):
        self.freq = freq_mhz
        self.tck = 1000.0 / freq_mhz  # ns per clock

        if dram_type == "DDR4":
            self.tRCD = int(13.75 / self.tck)   # ACT to READ/WRITE
            self.tRP = int(13.75 / self.tck)     # PRECHARGE period
            self.tRAS = int(32.0 / self.tck)     # ACT to PRE
            self.tRC = self.tRCD + self.tRP       # ACT to ACT
            self.tRTP = int(7.5 / self.tck)      # READ to PRE
            self.tFAW = int(28.0 / self.tck)     # 4-activate window
            self.tCCD = 4                        # CAS to CAS
            self.tRRD = int(4.875 / self.tck)    # ACT to ACT (different bank)
            self.tREFI = int(7800 / self.tck)    # refresh interval
            self.CL = int(14.0 / self.tck)       # CAS latency
            self.WL = int(14.0 / self.tck)       # write latency
        elif dram_type == "LPDDR4":
            self.tRCD = int(18.0 / self.tck)
            self.tRP = int(18.0 / self.tck)
            self.tRAS = int(42.0 / self.tck)
            self.tRC = self.tRCD + self.tRP
            self.tRTP = int(8.0 / self.tck)
            self.tFAW = int(32.0 / self.tck)
            self.tCCD = 4
            self.tRRD = int(6.0 / self.tck)
            self.tREFI = int(3900 / self.tck)
            self.CL = int(20.0 / self.tck)
            self.WL = int(20.0 / self.tck)
        else:
            self.tRCD = 15
            self.tRP = 15
            self.tRAS = 35
            self.tRC = 50
            self.tRTP = 8
            self.tFAW = 30
            self.tCCD = 4
            self.tRRD = 6
            self.tREFI = 7800
            self.CL = 14
            self.WL = 14


@dataclass
class BankState:
    bank_id: int
    active_row: Optional[int] = None
    last_activate_cycle: int = -999
    last_read_cycle: int = -999
    last_write_cycle: int = -999
    last_precharge_cycle: int = -999

    @property
    def is_open(self) -> bool:
        return self.active_row is not None


@dataclass
class MemoryRequest:
    address: int
    is_write: bool = False
    size_bytes: int = 64
    bank: int = 0
    row: int = 0
    col: int = 0
    issued_cycle: int = -1
    complete_cycle: int = -1


class MemoryController:
    """DRAM memory controller with command scheduling."""

    def __init__(self, n_banks: int = 8, dram_type: str = "DDR4",
                 freq_mhz: int = 1600, bus_width: int = 32):
        self.n_banks = n_banks
        self.timing = DRAMTiming(freq_mhz, dram_type)
        self.bus_width = bus_width  # bits
        self.banks = [BankState(i) for i in range(n_banks)]
        self.cycle = 0
        self.pending: List[MemoryRequest] = []
        self.completed: List[MemoryRequest] = []
        self.commands_issued: List[Dict] = []
        self.total_reads = 0
        self.total_writes = 0
        self.total_bytes = 0
        self.refresh_pending = 0
        self.last_refresh_cycle = 0

    def decode_address(self, addr: int) -> Tuple[int, int, int]:
        """Decode physical address to bank/row/col."""
        bank_bits = int(math.log2(self.n_banks))
        col_bits = 10
        row_bits = 16 - bank_bits - col_bits

        bank = (addr >> col_bits) & ((1 << bank_bits) - 1)
        row = (addr >> (col_bits + bank_bits)) & ((1 << row_bits) - 1)
        col = addr & ((1 << col_bits) - 1)
        return bank, row, col

    def enqueue(self, addr: int, is_write: bool = False, size: int = 64):
        bank, row, col = self.decode_address(addr)
        req = MemoryRequest(addr, is_write, size, bank, row, col)
        self.pending.append(req)

    def can_issue(self, cmd: DRAMCommand, bank_id: int) -> Tuple[bool, str]:
        """Check if command can be issued (timing constraints)."""
        bank = self.banks[bank_id]
        t = self.timing

        if cmd in (DRAMCommand.READ, DRAMCommand.WRITE):
            if not bank.is_open:
                return False, "bank not open"
            elapsed = self.cycle - bank.last_activate_cycle
            if elapsed < t.tRCD:
                return False, f"tRCD: need {t.tRCD}, have {elapsed}"

            if cmd == DRAMCommand.READ:
                elapsed = self.cycle - bank.last_write_cycle
                if elapsed < t.tCCD:
                    return False, f"tCCD write-to-read"

            elapsed = self.cycle - bank.last_read_cycle
            if elapsed < t.tCCD:
                return False, f"tCCD"

        elif cmd == DRAMCommand.ACTIVATE:
            elapsed = self.cycle - bank.last_precharge_cycle
            if elapsed < t.tRP:
                return False, f"tRP: need {t.tRP}, have {elapsed}"
            if bank.is_open:
                return False, "bank already open"

        elif cmd == DRAMCommand.PRECHARGE:
            if not bank.is_open:
                return False, "bank already closed"
            elapsed = self.cycle - bank.last_read_cycle
            if elapsed < t.tRTP:
                return False, f"tRTP: need {t.tRTP}, have {elapsed}"
            elapsed = self.cycle - bank.last_activate_cycle
            if elapsed < t.tRAS:
                return False, f"tRAS: need {t.tRAS}, have {elapsed}"

        elif cmd == DRAMCommand.REFRESH:
            elapsed = self.cycle - self.last_refresh_cycle
            if elapsed < t.tREFI:
                return False, f"tREFI not needed yet"

        return True, "OK"

    def step(self) -> int:
        """Advance one cycle. Returns number of commands issued."""
        issued = 0

        # Refresh check
        if self.cycle - self.last_refresh_cycle >= self.timing.tREFI:
            can, why = self.can_issue(DRAMCommand.REFRESH, 0)
            if can:
                self.commands_issued.append({"cycle": self.cycle, "cmd": "REFRESH"})
                self.last_refresh_cycle = self.cycle
                for b in self.banks:
                    b.last_precharge_cycle = self.cycle
                    b.active_row = None
                issued += 1
                self.cycle += 1
                return issued

        # Process pending requests
        remaining = []
        for req in self.pending:
            bank = self.banks[req.bank]

            # Need activate?
            if not bank.is_open or bank.active_row != req.row:
                if bank.is_open:
                    can, _ = self.can_issue(DRAMCommand.PRECHARGE, req.bank)
                    if can:
                        self.commands_issued.append({"cycle": self.cycle, "cmd": "PRE", "bank": req.bank})
                        bank.last_precharge_cycle = self.cycle
                        bank.active_row = None
                        issued += 1
                        self.cycle += 1
                    remaining.append(req)
                    continue

                can, _ = self.can_issue(DRAMCommand.ACTIVATE, req.bank)
                if can:
                    self.commands_issued.append({"cycle": self.cycle, "cmd": "ACT",
                                                 "bank": req.bank, "row": req.row})
                    bank.last_activate_cycle = self.cycle
                    bank.active_row = req.row
                    issued += 1
                    self.cycle += 1
                remaining.append(req)
                continue

            # Issue read/write
            cmd = DRAMCommand.WRITE if req.is_write else DRAMCommand.READ
            can, _ = self.can_issue(cmd, req.bank)
            if can:
                latency = self.timing.CL if not req.is_write else self.timing.WL
                req.issued_cycle = self.cycle
                req.complete_cycle = self.cycle + latency
                self.commands_issued.append({"cycle": self.cycle,
                                             "cmd": "WR" if req.is_write else "RD",
                                             "bank": req.bank, "col": req.col})
                if req.is_write:
                    bank.last_write_cycle = self.cycle
                    self.total_writes += 1
                else:
                    bank.last_read_cycle = self.cycle
                    self.total_reads += 1
                self.total_bytes += req.size_bytes
                issued += 1
                self.cycle += 1
            else:
                remaining.append(req)

        self.pending = remaining

        # Check completed
        done = [r for r in self.pending if r.complete_cycle > 0 and self.cycle >= r.complete_cycle]
        self.completed.extend(done)
        for d in done:
            if d in self.pending:
                self.pending.remove(d)

        self.cycle += 1
        return issued


def demo():
    print("=== Memory Controller Simulator ===\n")

    mc = MemoryController(n_banks=8, dram_type="DDR4", freq_mhz=1600)
    print(f"DRAM: DDR4 @ {mc.timing.freq}MHz, tCK={mc.timing.tck:.2f}ns")
    print(f"  tRCD={mc.timing.tRCD}, tRP={mc.timing.tRP}, tRAS={mc.timing.tRAS}")
    print(f"  CL={mc.timing.CL}, tREFI={mc.timing.tREFI}")
    print(f"  Banks: {mc.n_banks}, Bus: {mc.bus_width}bit")
    print()

    # Sequential reads (same bank)
    print("--- Sequential Reads (Bank 0) ---")
    for i in range(16):
        mc.enqueue(i * 64, False, 64)

    start_cycle = mc.cycle
    while mc.pending:
        mc.step()
    end_cycle = mc.cycle

    elapsed_ns = (end_cycle - start_cycle) * mc.timing.tck
    bw = mc.total_bytes / (elapsed_ns * 1e-9) / 1e9
    print(f"  16 reads, {end_cycle - start_cycle} cycles, {elapsed_ns:.1f}ns")
    print(f"  Bandwidth: {bw:.2f} GB/s")
    print()

    # Interleaved reads (across banks)
    mc2 = MemoryController(8, "DDR4", 1600, 32)
    print("--- Interleaved Reads (8 banks) ---")
    for i in range(16):
        bank = i % 8
        mc2.enqueue((bank << 10) + (i // 8) * 64, False, 64)

    start_cycle = mc2.cycle
    while mc2.pending:
        mc2.step()
    end_cycle = mc2.cycle

    elapsed_ns = (end_cycle - start_cycle) * mc2.timing.tck
    bw = mc2.total_bytes / (elapsed_ns * 1e-9) / 1e9
    print(f"  16 reads, {end_cycle - start_cycle} cycles, {elapsed_ns:.1f}ns")
    print(f"  Bandwidth: {bw:.2f} GB/s")
    print(f"  Speedup: {bw / 0.01:.1f}x" if bw > 0 else "")
    print()

    # Weight streaming simulation
    print("--- Weight Streaming (1MB from 8 banks) ---")
    mc3 = MemoryController(8, "LPDDR4", 2133, 32)
    chunk_size = 256  # bytes per transfer
    n_chunks = 1024 * 1024 // chunk_size
    for i in range(n_chunks):
        bank = i % 8
        mc3.enqueue((bank << 10) + (i // 8) * chunk_size, False, chunk_size)

    total_bytes = n_chunks * chunk_size
    start_cycle = mc3.cycle
    while mc3.pending:
        mc3.step()
        if mc3.cycle - start_cycle > 100000:
            break
    end_cycle = mc3.cycle

    actual_bytes = mc3.total_bytes
    elapsed_ns = (end_cycle - start_cycle) * mc3.timing.tck
    if elapsed_ns > 0:
        bw = actual_bytes / (elapsed_ns * 1e-9) / 1e9
    else:
        bw = 0
    print(f"  Requested: {total_bytes/1024:.0f}KB, Completed: {actual_bytes/1024:.0f}KB")
    print(f"  Cycles: {end_cycle - start_cycle}, Time: {elapsed_ns:.0f}ns")
    print(f"  Bandwidth: {bw:.2f} GB/s")


if __name__ == "__main__":
    demo()
