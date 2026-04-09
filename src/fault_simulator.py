#!/usr/bin/env python3
"""Fault injection and simulation for mask-locked chips.

Simulates stuck-at faults, bridging faults, and delay faults
to verify chip resilience and yield impact.
"""
import random, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class FaultType(Enum):
    STUCK_AT_0 = "SA0"
    STUCK_AT_1 = "SA1"
    STUCK_OPEN = "SOPEN"
    BRIDGE_AND = "BR_AND"
    BRIDGE_OR = "BR_OR"
    DELAY = "DELAY"


class FaultLocation(Enum):
    GATE_OUTPUT = "gate_out"
    GATE_INPUT = "gate_in"
    WIRE = "wire"
    REGISTER = "register"
    MEMORY = "memory"


@dataclass
class Fault:
    fault_type: FaultType
    location: FaultLocation
    target: str  # signal/node name
    value: int = 0  # for stuck-at
    bridge_target: str = ""  # for bridging
    delay_ps: float = 0  # for delay faults
    active: bool = True


@dataclass
class FaultResult:
    fault: Fault
    detected: bool
    detected_by: str = ""
    cycles_to_detect: int = 0
    circuit_still_functional: bool = True


class FaultSimulator:
    """Inject faults and simulate their effects."""

    def __init__(self, n_gates: int = 1000, n_wires: int = 2000,
                 n_registers: int = 100):
        self.n_gates = n_gates
        self.n_wires = n_wires
        self.n_registers = n_registers
        self.signals: Dict[str, int] = {}
        self.faults: List[Fault] = []
        self.results: List[FaultResult] = []

        # Initialize signals
        for i in range(n_gates):
            self.signals[f"g_{i}_out"] = 0
        for i in range(n_wires):
            self.signals[f"w_{i}"] = 0
        for i in range(n_registers):
            self.signals[f"r_{i}"] = 0

        # Create scan chain register order
        self.scan_chain = [f"r_{i}" for i in range(n_registers)]

    def inject_random_faults(self, n_faults: int = 10,
                              fault_types: Optional[List[FaultType]] = None,
                              locations: Optional[List[FaultLocation]] = None) -> List[Fault]:
        """Inject random faults."""
        if fault_types is None:
            fault_types = [FaultType.STUCK_AT_0, FaultType.STUCK_AT_1,
                          FaultType.STUCK_AT_0, FaultType.STUCK_AT_1,
                          FaultType.DELAY]

        if locations is None:
            locations = [FaultLocation.GATE_OUTPUT, FaultLocation.WIRE,
                        FaultLocation.GATE_OUTPUT, FaultLocation.WIRE,
                        FaultLocation.REGISTER]

        targets = (list(self.signals.keys()))

        for _ in range(n_faults):
            ft = random.choice(fault_types)
            loc = random.choice(locations)
            target = random.choice(targets)
            val = random.randint(0, 1)

            fault = Fault(ft, loc, target, value=val)
            if ft == FaultType.DELAY:
                fault.delay_ps = random.uniform(10, 200)
            elif ft in (FaultType.BRIDGE_AND, FaultType.BRIDGE_OR):
                other = random.choice([t for t in targets if t != target])
                fault.bridge_target = other

            self.faults.append(fault)
        return self.faults

    def inject_faults(self, faults: List[Fault]):
        self.faults.extend(faults)

    def _apply_faults(self) -> Dict[str, int]:
        """Apply active faults to signals."""
        corrupted = dict(self.signals)
        for fault in self.faults:
            if not fault.active or fault.target not in corrupted:
                continue
            if fault.fault_type == FaultType.STUCK_AT_0:
                corrupted[fault.target] = 0
            elif fault.fault_type == FaultType.STUCK_AT_1:
                corrupted[fault.target] = 1
            elif fault.fault_type == FaultType.STUCK_OPEN:
                corrupted[fault.target] = 0  # floating → pulled low
            elif fault.fault_type == FaultType.BRIDGE_AND:
                if fault.bridge_target in corrupted:
                    corrupted[fault.target] &= corrupted[fault.bridge_target]
            elif fault.fault_type == FaultType.BRIDGE_OR:
                if fault.bridge_target in corrupted:
                    corrupted[fault.target] |= corrupted[fault.bridge_target]
        return corrupted

    def simulate_cycle(self, input_values: Dict[str, int] = None) -> Dict[str, int]:
        """Simulate one cycle with fault injection."""
        if input_values:
            self.signals.update(input_values)

        # Random signal activity
        for k in self.signals:
            if k not in (input_values or {}):
                if random.random() < 0.3:
                    self.signals[k] = random.randint(0, 1)

        return self._apply_faults()

    def run_scan_test(self) -> List[FaultResult]:
        """Run scan-based test pattern generation."""
        self.results = []

        # Initialize with random patterns
        for i in range(10):
            pattern = {f"r_{j}": random.randint(0, 1) for j in range(self.n_registers)}
            self.simulate_cycle(pattern)

        # Scan out and compare
        observed = self._apply_faults()

        for fault in self.faults:
            # Check if fault affects scan chain output
            detected = False
            detector = ""

            if fault.target in self.scan_chain:
                expected = self.signals.get(fault.target, 0)
                actual = observed.get(fault.target, 0)
                if expected != actual:
                    detected = True
                    detector = f"scan_out:{fault.target}"

            if not detected and fault.fault_type in (FaultType.BRIDGE_AND, FaultType.BRIDGE_OR):
                if fault.bridge_target in observed:
                    detected = True
                    detector = f"bridge:{fault.bridge_target}"

            # Check functional impact
            functional = True
            for sig_name in ["g_0_out", "r_0"]:  # critical signals
                if fault.target == sig_name:
                    functional = False

            self.results.append(FaultResult(
                fault=fault, detected=detected,
                detected_by=detector,
                circuit_still_functional=functional))

        return self.results

    def fault_coverage(self) -> Dict:
        """Calculate fault coverage metrics."""
        if not self.results:
            self.run_scan_test()

        total = len(self.results)
        detected = sum(1 for r in self.results if r.detected)
        functional = sum(1 for r in self.results if r.circuit_still_functional)

        by_type = {}
        for r in self.results:
            ft = r.fault.fault_type.value
            if ft not in by_type:
                by_type[ft] = {"total": 0, "detected": 0}
            by_type[ft]["total"] += 1
            if r.detected:
                by_type[ft]["detected"] += 1

        return {
            "total_faults": total,
            "detected": detected,
            "coverage_pct": round(detected / total * 100, 1) if total > 0 else 0,
            "functional_pct": round(functional / total * 100, 1) if total > 0 else 0,
            "by_type": by_type,
        }


def demo():
    print("=== Fault Simulator ===\n")

    sim = FaultSimulator(n_gates=500, n_wires=1000, n_registers=50)

    # Stuck-at faults
    print("--- Stuck-At Faults (10 SA0/SA1) ---")
    faults = sim.inject_random_faults(10, [FaultType.STUCK_AT_0, FaultType.STUCK_AT_1])
    results = sim.run_scan_test()
    cov = sim.fault_coverage()
    print(f"  Faults: {cov['total_faults']}, Detected: {cov['detected']}, "
          f"Coverage: {cov['coverage_pct']}%")
    print(f"  Functional: {cov['functional_pct']}%")
    print()

    # Mixed fault types
    print("--- Mixed Faults (20) ---")
    sim2 = FaultSimulator(500, 1000, 50)
    sim2.inject_random_faults(20)
    results2 = sim2.run_scan_test()
    cov2 = sim2.fault_coverage()
    print(f"  Faults: {cov2['total_faults']}, Detected: {cov2['detected']}, "
          f"Coverage: {cov2['coverage_pct']}%")
    print(f"  By type:")
    for ft, stats in cov2["by_type"].items():
        pct = stats["detected"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {ft:10s}: {stats['detected']}/{stats['total']} ({pct:.0f}%)")
    print()

    # Yield impact
    print("--- Yield Impact vs Fault Density ---")
    for fault_rate in [0.001, 0.005, 0.01, 0.02, 0.05]:
        sim3 = FaultSimulator(500, 1000, 50)
        n_faults = max(1, int(500 * 1000 * fault_rate))
        sim3.inject_random_faults(n_faults)
        sim3.run_scan_test()
        cov = sim3.fault_coverage()
        good = cov["functional_pct"] / 100
        print(f"  {fault_rate*100:.1f}% defect rate: {n_faults:3d} faults, "
              f"{good*100:.1f}% functional chips")

    # Swarm tiling fault tolerance
    print("\n--- Swarm Tiling Fault Tolerance ---")
    for n_tiles in [4, 8, 16, 32, 64]:
        faults_in_tiles = max(0, int(n_tiles * 0.05))  # 5% tile defect rate
        working = n_tiles - faults_in_tiles
        capacity_pct = working / n_tiles * 100
        print(f"  {n_tiles:2d} tiles: {faults_in_tiles} defective, "
              f"{working} working ({capacity_pct:.0f}% capacity)")


if __name__ == "__main__":
    demo()
