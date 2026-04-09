#!/usr/bin/env python3
"""RTL simulation engine for mask-locked chip netlists.

Event-driven simulator supporting combinational logic,
sequential elements (DFFs), and basic arithmetic.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum


class SignalState(Enum):
    ZERO = 0
    ONE = 1
    X = 2  # unknown
    Z = 3  # high-impedance


class SimSignal:
    """Simulation signal (multi-bit)."""
    _next_id = 0

    def __init__(self, name: str, width: int = 1, value: int = 0):
        self.id = SimSignal._next_id
        SimSignal._next_id += 1
        self.name = name
        self.width = width
        self.mask = (1 << width) - 1
        self.value = value & self.mask
        self.drivers: List[Tuple['SimPrimitive', str]] = []

    def set(self, value: int):
        self.value = value & self.mask

    def bit(self, idx: int) -> int:
        return (self.value >> idx) & 1

    def __repr__(self):
        if self.width == 1:
            return f"{self.name}={self.value}"
        return f"{self.name}={self.value:0{self.width}b}"


class SimPrimitive:
    """Base simulation primitive."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, inputs: Dict[str, int]) -> Dict[str, int]:
        raise NotImplementedError


class ANDGate(SimPrimitive):
    def __init__(self, name: str, width: int = 1):
        super().__init__(name)
        self.width = width

    def evaluate(self, inputs: Dict[str, int]) -> Dict[str, int]:
        a = inputs.get("A", 0)
        b = inputs.get("B", 0)
        return {"Y": a & b}


class ORGate(SimPrimitive):
    def __init__(self, name: str, width: int = 1):
        super().__init__(name)
        self.width = width

    def evaluate(self, inputs: Dict[str, int]) -> Dict[str, int]:
        return {"Y": inputs.get("A", 0) | inputs.get("B", 0)}


class XORGate(SimPrimitive):
    def __init__(self, name: str, width: int = 1):
        super().__init__(name)
        self.width = width

    def evaluate(self, inputs: Dict[str, int]) -> Dict[str, int]:
        return {"Y": inputs.get("A", 0) ^ inputs.get("B", 0)}


class INVGate(SimPrimitive):
    def evaluate(self, inputs: Dict[str, int]) -> Dict[str, int]:
        mask = (1 << self.width) - 1 if hasattr(self, 'width') else 1
        return {"Y": (~inputs.get("A", 0)) & mask}


class MUX2(SimPrimitive):
    def evaluate(self, inputs: Dict[str, int]) -> Dict[str, int]:
        s = inputs.get("S", 0) & 1
        return {"Y": inputs.get("B", 0) if s else inputs.get("A", 0)}


class DFlipFlop(SimPrimitive):
    """Edge-triggered D flip-flop."""

    def __init__(self, name: str, width: int = 1):
        super().__init__(name)
        self.width = width
        self.q_value = 0
        self.last_clk = 0

    def evaluate(self, inputs: Dict[str, int]) -> Dict[str, int]:
        clk = inputs.get("CK", 0) & 1
        rn = inputs.get("RN", 1) & 1

        if rn == 0:
            self.q_value = 0
        elif clk == 1 and self.last_clk == 0:  # rising edge
            d = inputs.get("D", 0) & ((1 << self.width) - 1)
            self.q_value = d

        self.last_clk = clk
        return {"Q": self.q_value}


class FullAdder(SimPrimitive):
    def evaluate(self, inputs: Dict[str, int]) -> Dict[str, int]:
        a = inputs.get("A", 0) & 1
        b = inputs.get("B", 0) & 1
        ci = inputs.get("CI", 0) & 1
        s = a ^ b ^ ci
        co = (a & b) | (a & ci) | (b & ci)
        return {"S": s, "CO": co}


class Simulator:
    """Event-driven RTL simulator."""

    def __init__(self):
        self.signals: Dict[str, SimSignal] = {}
        self.primitives: List[SimPrimitive] = []
        self.connections: Dict[str, Dict[str, str]] = {}  # prim_name -> {port: signal_name}
        self.cycle = 0
        self.trace: List[Dict[str, int]] = []
        self.max_trace = 1000

    def add_signal(self, name: str, width: int = 1, value: int = 0) -> SimSignal:
        sig = SimSignal(name, width, value)
        self.signals[name] = sig
        return sig

    def add_primitive(self, prim: SimPrimitive, connections: Dict[str, str]):
        self.primitives.append(prim)
        self.connections[prim.name] = connections

    def get_signal(self, name: str) -> SimSignal:
        return self.signals[name]

    def set_input(self, name: str, value: int):
        if name in self.signals:
            self.signals[name].set(value)

    def read_output(self, name: str) -> int:
        return self.signals.get(name, SimSignal(name, 1)).value

    def _resolve_inputs(self, prim: SimPrimitive) -> Dict[str, int]:
        conns = self.connections.get(prim.name, {})
        result = {}
        for port, sig_name in conns.items():
            if sig_name in self.signals:
                result[port] = self.signals[sig_name].value
        return result

    def _apply_outputs(self, prim: SimPrimitive, outputs: Dict[str, int]):
        conns = self.connections.get(prim.name, {})
        for port, sig_name in conns.items():
            if port in outputs and sig_name in self.signals:
                self.signals[sig_name].set(outputs[port])

    def step(self, clk_signal: str = "clk") -> Dict:
        """Advance one clock cycle."""
        self.cycle += 1
        # Drive clock high
        if clk_signal in self.signals:
            self.signals[clk_signal].set(1)

        # Evaluate all primitives
        for prim in self.primitives:
            inputs = self._resolve_inputs(prim)
            outputs = prim.evaluate(inputs)
            self._apply_outputs(prim, outputs)

        # Drive clock low
        if clk_signal in self.signals:
            self.signals[clk_signal].set(0)
            # Re-evaluate for falling edge (DFFs sample on rising)
            for prim in self.primitives:
                inputs = self._resolve_inputs(prim)
                outputs = prim.evaluate(inputs)
                self._apply_outputs(prim, outputs)

        # Trace
        if len(self.trace) < self.max_trace:
            snapshot = {name: sig.value for name, sig in self.signals.items()}
            self.trace.append(snapshot)

        return {name: sig.value for name, sig in self.signals.items()}

    def run(self, cycles: int, clk: str = "clk") -> List[Dict]:
        results = []
        for _ in range(cycles):
            state = self.step(clk)
            results.append(state)
        return results


def demo():
    print("=== RTL Simulation Engine ===\n")

    sim = Simulator()

    # Signals
    sim.add_signal("clk", 1)
    sim.add_signal("rst_n", 1, 1)
    sim.add_signal("a", 4, 3)
    sim.add_signal("b", 4, 5)
    sim.add_signal("and_out", 4)
    sim.add_signal("or_out", 4)
    sim.add_signal("xor_out", 4)
    sim.add_signal("sum_out", 4)
    sim.add_signal("carry_out", 1)
    sim.add_signal("reg_out", 4)
    sim.add_signal("mux_out", 4)
    sim.add_signal("sel", 1, 0)

    # Combinational logic
    and_gate = ANDGate("and1", 4)
    sim.add_primitive(and_gate, {"A": "a", "B": "b", "Y": "and_out"})

    or_gate = ORGate("or1", 4)
    sim.add_primitive(or_gate, {"A": "a", "B": "b", "Y": "or_out"})

    xor_gate = XORGate("xor1", 4)
    sim.add_primitive(xor_gate, {"A": "a", "B": "b", "Y": "xor_out"})

    fa = FullAdder("fa1")
    sim.add_primitive(fa, {"A": "a", "B": "b", "CI": "carry_out", "S": "sum_out", "CO": "carry_out"})

    mux = MUX2("mux1")
    sim.add_primitive(mux, {"A": "and_out", "B": "or_out", "S": "sel", "Y": "mux_out"})

    # Sequential
    dff = DFlipFlop("reg1", 4)
    sim.add_primitive(dff, {"D": "mux_out", "CK": "clk", "RN": "rst_n", "Q": "reg_out"})

    # Test
    print("--- Combinational Logic ---")
    sim.set_input("a", 3)
    sim.set_input("b", 5)
    sim.step("clk")
    print(f"  a={sim.read_output('a')}, b={sim.read_output('b')}")
    print(f"  AND={sim.read_output('and_out')} ({3 & 5})")
    print(f"  OR ={sim.read_output('or_out')} ({3 | 5})")
    print(f"  XOR={sim.read_output('xor_out')} ({3 ^ 5})")
    print()

    # MUX
    print("--- MUX ---")
    sim.set_input("sel", 0)
    sim.step("clk")
    print(f"  sel=0: mux={sim.read_output('mux_out')} (should be AND={3 & 5})")
    sim.set_input("sel", 1)
    sim.step("clk")
    print(f"  sel=1: mux={sim.read_output('mux_out')} (should be OR={3 | 5})")
    print()

    # Sequential
    print("--- D Flip-Flop Pipeline ---")
    for i in range(5):
        sim.set_input("a", i)
        sim.set_input("b", i * 2)
        sim.step("clk")
        print(f"  Cycle {sim.cycle}: a={i}, b={i*2}, dff_out={sim.read_output('reg_out')}")
    print()

    # Multi-cycle counter
    print("--- 4-bit Counter ---")
    sim2 = Simulator()
    sim2.add_signal("clk", 1)
    sim2.add_signal("rst_n", 1, 1)
    sim2.add_signal("count", 4, 0)
    sim2.add_signal("count_next", 4)

    fa_inc = FullAdder("inc_fa")
    sim2.add_primitive(fa_inc, {"A": "count", "B": "1", "CI": "0", "S": "count_next", "CO": "dummy"})
    sim2.add_signal("dummy", 1)

    dff2 = DFlipFlop("count_dff", 4)
    sim2.add_primitive(dff2, {"D": "count_next", "CK": "clk", "RN": "rst_n", "Q": "count"})

    print("  Count: ", end="")
    for _ in range(20):
        sim2.step("clk")
        print(f"{sim2.read_output('count')} ", end="")
    print()


if __name__ == "__main__":
    demo()
