#!/usr/bin/env python3
"""JTAG debug interface for mask-locked inference chips.

TAP controller, instruction register, debug access port,
and boundary scan for chip bring-up and production test.
"""
import struct, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import IntEnum


class TAPState(IntEnum):
    """IEEE 1149.1 TAP controller states."""
    RESET = 0
    IDLE = 1
    SELECT_DR = 2
    CAPTURE_DR = 3
    SHIFT_DR = 4
    EXIT1_DR = 5
    PAUSE_DR = 6
    EXIT2_DR = 7
    UPDATE_DR = 8
    SELECT_IR = 9
    CAPTURE_IR = 10
    SHIFT_IR = 11
    EXIT1_IR = 12
    PAUSE_IR = 13
    EXIT2_IR = 14
    UPDATE_IR = 15


# TAP state transitions (TMS=0, TMS=1)
TAP_TRANSITIONS = {
    TAPState.RESET: (TAPState.IDLE, TAPState.RESET),
    TAPState.IDLE: (TAPState.IDLE, TAPState.SELECT_DR),
    TAPState.SELECT_DR: (TAPState.CAPTURE_DR, TAPState.SELECT_IR),
    TAPState.CAPTURE_DR: (TAPState.SHIFT_DR, TAPState.EXIT1_DR),
    TAPState.SHIFT_DR: (TAPState.SHIFT_DR, TAPState.EXIT1_DR),
    TAPState.EXIT1_DR: (TAPState.PAUSE_DR, TAPState.UPDATE_DR),
    TAPState.PAUSE_DR: (TAPState.PAUSE_DR, TAPState.EXIT2_DR),
    TAPState.EXIT2_DR: (TAPState.SHIFT_DR, TAPState.UPDATE_DR),
    TAPState.UPDATE_DR: (TAPState.IDLE, TAPState.SELECT_DR),
    TAPState.SELECT_IR: (TAPState.CAPTURE_IR, TAPState.RESET),
    TAPState.CAPTURE_IR: (TAPState.SHIFT_IR, TAPState.EXIT1_IR),
    TAPState.SHIFT_IR: (TAPState.SHIFT_IR, TAPState.EXIT1_IR),
    TAPState.EXIT1_IR: (TAPState.PAUSE_IR, TAPState.UPDATE_IR),
    TAPState.PAUSE_IR: (TAPState.PAUSE_IR, TAPState.EXIT2_IR),
    TAPState.EXIT2_IR: (TAPState.SHIFT_IR, TAPState.UPDATE_IR),
    TAPState.UPDATE_IR: (TAPState.IDLE, TAPState.SELECT_DR),
}


class JTAGInstruction:
    def __init__(self, name: str, opcode: int, data_width: int):
        self.name = name
        self.opcode = opcode
        self.data_width = data_width


class TAPController:
    """IEEE 1149.1 Test Access Port controller."""

    def __init__(self, ir_width: int = 4):
        self.ir_width = ir_width
        self.state = TAPState.RESET
        self.ir = 0
        self.dr = 0
        self.ir_shift_reg = 0
        self.dr_shift_reg = 0
        self.tdo = 0
        self.instructions: Dict[str, JTAGInstruction] = {
            "BYPASS": JTAGInstruction("BYPASS", 0xF, 1),
            "EXTEST": JTAGInstruction("EXTEST", 0x0, 0),
            "SAMPLE": JTAGInstruction("SAMPLE", 0x2, 0),
            "IDCODE": JTAGInstruction("IDCODE", 0x1, 32),
        }

    def add_instruction(self, name: str, opcode: int, data_width: int):
        self.instructions[name] = JTAGInstruction(name, opcode, data_width)

    def _transition(self, tms: int):
        tms0, tms1 = TAP_TRANSITIONS[self.state]
        self.state = tms1 if tms else tms0

    def reset(self):
        """5 TMS=1 cycles to guarantee reset."""
        for _ in range(5):
            self._transition(1)
        return self.state == TAPState.RESET

    def go_idle(self):
        while self.state != TAPState.IDLE:
            if self.state == TAPState.RESET:
                self._transition(0)
            elif self.state in (TAPState.SELECT_DR, TAPState.SELECT_IR):
                self._transition(0)
            else:
                self._transition(1)

    def shift_ir(self, instruction_name: str) -> int:
        """Shift instruction into IR."""
        instr = self.instructions.get(instruction_name)
        if not instr:
            return -1
        self.go_idle()
        self._transition(1)  # -> SELECT_DR
        self._transition(1)  # -> SELECT_IR
        self._transition(0)  # -> CAPTURE_IR
        self._transition(0)  # -> SHIFT_IR

        opcode = instr.opcode
        tdo_bits = 0
        for i in range(self.ir_width):
            bit = (opcode >> i) & 1
            self.tdo = (self.ir_shift_reg >> i) & 1
            tdo_bits |= self.tdo << i
            if i == self.ir_width - 1:
                self._transition(1)  # -> EXIT1_IR on last bit
            else:
                self._transition(0)  # stay in SHIFT_IR

        self._transition(1)  # -> UPDATE_IR
        self._transition(0)  # -> IDLE
        self.ir = opcode & ((1 << self.ir_width) - 1)
        return tdo_bits

    def shift_dr(self, data: int, num_bits: int = 32) -> int:
        """Shift data through DR. Returns TDO."""
        self.go_idle()
        self._transition(1)  # -> SELECT_DR
        self._transition(0)  # -> CAPTURE_DR
        self._transition(0)  # -> SHIFT_DR

        tdo_bits = 0
        for i in range(num_bits):
            self.tdo = (self.dr_shift_reg >> i) & 1
            tdo_bits |= self.tdo << i
            if i == num_bits - 1:
                self._transition(1)  # EXIT1_DR
            else:
                self._transition(0)  # stay SHIFT_DR

        self._transition(1)  # UPDATE_DR
        self._transition(0)  # IDLE
        self.dr = data & ((1 << num_bits) - 1)
        return tdo_bits


class DebugAccessPort:
    """ARM DAP-like debug access for chip internals."""

    def __init__(self, tap: TAPController):
        self.tap = tap
        tap.add_instruction("DEBUG", 0x4, 35)
        self.registers: Dict[str, int] = {
            "DP_CTRL": 0x0,
            "DP_STAT": 0x0,
            "DP_SELECT": 0x0,
            "DP_RDBUFF": 0x0,
        }
        self.debug_ir_active = False

    def _ensure_debug_ir(self):
        if not self.debug_ir_active:
            self.tap.shift_ir("DEBUG")
            self.debug_ir_active = True

    def write_reg(self, addr: int, data: int) -> int:
        """Write debug register. Returns ACK."""
        self._ensure_debug_ir()
        payload = (1 << 32) | ((addr & 0xF) << 28) | (data & 0xFFFFFFF)
        tdo = self.tap.shift_dr(payload, 35)
        ack = (tdo >> 33) & 0x3
        return ack

    def read_reg(self, addr: int) -> Tuple[int, int]:
        """Read debug register. Returns (ack, data)."""
        self._ensure_debug_ir()
        payload = (addr & 0xF) << 28
        tdo = self.tap.shift_dr(payload, 35)
        ack = (tdo >> 33) & 0x3
        data = tdo & 0xFFFFFFF
        return ack, data

    def halt_core(self):
        """Halt processor via debug request."""
        return self.write_reg(0x0, 0x1)  # set debug halt

    def resume_core(self):
        """Resume processor."""
        return self.write_reg(0x0, 0x0)  # clear debug halt

    def read_memory(self, addr: int) -> int:
        """Read memory via debug access."""
        self.write_reg(0x4, addr)  # TAR = addr
        ack, data = self.read_reg(0xC)  # DRW
        return data


class BoundaryScan:
    """IEEE 1149.1 boundary scan for I/O testing."""

    def __init__(self, n_pins: int = 40):
        self.n_pins = n_pins
        self.capture_reg = 0
        self.update_reg = 0
        self.boundary_cells = [{"capture": 0, "update": 0, "enable": 0}
                               for _ in range(n_pins * 2)]  # input + output per pin

    def capture(self, pin_values: Dict[int, int]):
        """Capture current pin values."""
        reg = 0
        for pin, val in pin_values.items():
            self.boundary_cells[pin]["capture"] = val & 1
            reg |= (val & 1) << pin
        self.capture_reg = reg

    def shift_update(self, pattern: int):
        """Shift pattern into boundary cells and update outputs."""
        for i in range(self.n_pins * 2):
            bit = (pattern >> i) & 1
            self.boundary_cells[i]["update"] = bit

    def get_output_values(self) -> Dict[int, int]:
        return {i: c["update"] for i, c in enumerate(self.boundary_cells)
                if c["enable"]}


def demo():
    print("=== JTAG Debug Interface ===\n")

    # TAP Controller
    print("--- TAP Controller ---")
    tap = TAPController(ir_width=4)
    tap.add_instruction("CUSTOM", 0x3, 16)

    print(f"  Initial state: {tap.state.name}")
    tap.reset()
    print(f"  After reset: {tap.state.name}")
    tap.go_idle()
    print(f"  After go_idle: {tap.state.name}")
    print()

    # IDCODE
    print("--- IDCODE ---")
    tap.shift_ir("IDCODE")
    idcode = 0x1AF41337  # vendor=1AF4, part=1337
    tdo = tap.shift_dr(idcode, 32)
    print(f"  Shifted IDCODE: 0x{idcode:08X}")
    print(f"  TDO: 0x{tdo:08X}")
    print()

    # Custom instruction
    print("--- Custom Instruction ---")
    tap.shift_ir("CUSTOM")
    data_out = tap.shift_dr(0xABCD, 16)
    print(f"  Shifted 0xABCD, TDO: 0x{data_out:04X}")
    print()

    # DAP
    print("--- Debug Access Port ---")
    dap = DebugAccessPort(tap)
    ack = dap.halt_core()
    print(f"  Halt core: ACK={ack}")
    ack = dap.write_reg(0x4, 0x20000000)  # write address
    print(f"  Write addr 0x20000000: ACK={ack}")
    ack, data = dap.read_reg(0xC)  # read data
    print(f"  Read data: ACK={ack}, DATA=0x{data:07X}")
    ack = dap.resume_core()
    print(f"  Resume core: ACK={ack}")
    print()

    # Boundary Scan
    print("--- Boundary Scan ---")
    bs = BoundaryScan(8)
    bs.capture({0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0})
    print(f"  Captured: 0b{bs.capture_reg:08b}")

    test_pattern = 0xAA
    bs.shift_update(test_pattern)
    print(f"  Update pattern: 0b{test_pattern:08b}")
    outputs = bs.get_output_values()
    print(f"  Output pins: {outputs}")
    print()

    # All instructions
    print("--- Available Instructions ---")
    for name, instr in tap.instructions.items():
        print(f"  {name:10s}: opcode=0x{instr.opcode:X}, width={instr.data_width}bit")


if __name__ == "__main__":
    demo()
