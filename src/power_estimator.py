#!/usr/bin/env python3
"""Power estimator for mask-locked inference chips.

Dynamic, static, and leakage power models based on activity factor,
switching capacitance, and process technology.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class ProcessNode:
    name: str
    nm: int
    vdd: float           # volts
    vt: float            # threshold voltage
    gate_cap_ff: float   # fF/um gate length
    leakage_nw: float    # nW/um at 25C
    oxide_nm: float      # gate oxide thickness

    @staticmethod
    def from_nm(nm: int) -> 'ProcessNode':
        nodes = {
            28: ProcessNode("TSMC 28LP", 28, 0.9, 0.45, 1.0, 0.5, 1.2),
            40: ProcessNode("TSMC 40LP", 40, 1.1, 0.50, 1.3, 0.2, 1.8),
            65: ProcessNode("TSMC 65LP", 65, 1.2, 0.55, 1.5, 0.05, 2.5),
            130: ProcessNode("TSMC 130G", 130, 1.5, 0.60, 2.0, 0.01, 3.5),
            7: ProcessNode("TSMC N7", 7, 0.75, 0.30, 0.7, 5.0, 0.8),
            5: ProcessNode("TSMC N5", 5, 0.70, 0.28, 0.6, 10.0, 0.7),
        }
        # Pick closest
        if nm in nodes:
            return nodes[nm]
        closest = min(nodes.keys(), key=lambda x: abs(x - nm))
        return nodes[closest]


@dataclass
class PowerBlock:
    name: str
    gate_count: int       # number of equivalent gates
    activity: float       # 0.0 to 1.0 switching activity
    clock_freq_mhz: float
    avg_fanout: float = 3.0

    @property
    def switching_events_per_sec(self) -> float:
        return self.gate_count * self.activity * self.clock_freq_mhz * 1e6


class DynamicPowerModel:
    """P_dynamic = alpha * C * V^2 * f"""

    def __init__(self, process: ProcessNode):
        self.process = process

    def gate_cap(self, fanout: float = 3.0) -> float:
        """Effective load capacitance per gate (fF)."""
        gate_len = self.process.nm / 1000.0  # nm to um
        wire_cap = fanout * gate_len * 0.2  # simplified wire cap
        return self.process.gate_cap_ff + wire_cap

    def block_power_mw(self, block: PowerBlock) -> float:
        cap = self.gate_cap(block.avg_fanout) * 1e-15  # fF to F
        alpha = block.activity
        v = self.process.vdd
        f = block.clock_freq_mhz * 1e6
        power = alpha * cap * v * v * f * block.gate_count
        return power * 1000  # W to mW


class LeakagePowerModel:
    """P_leakage based on subthreshold and gate leakage."""

    def __init__(self, process: ProcessNode, temp_c: float = 85.0):
        self.process = process
        self.temp = temp_c

    def temp_factor(self) -> float:
        """Leakage doubles every ~10C above 25C."""
        return 2 ** ((self.temp - 25) / 10)

    def block_leakage_mw(self, block: PowerBlock) -> float:
        gate_len = self.process.nm / 1000.0
        leakage_per_gate = self.process.leakage_nw * gate_len * self.temp_factor()
        return block.gate_count * leakage_per_gate * 1e-6  # nW to mW


class ChipPowerEstimator:
    """Full chip power estimation."""

    def __init__(self, process_nm: int = 28, temp_c: float = 85.0):
        self.process = ProcessNode.from_nm(process_nm)
        self.temp = temp_c
        self.dynamic = DynamicPowerModel(self.process)
        self.leakage = LeakagePowerModel(self.process, temp_c)
        self.blocks: List[PowerBlock] = []

    def add_block(self, name: str, gate_count: int, activity: float,
                  clock_mhz: float, fanout: float = 3.0):
        self.blocks.append(PowerBlock(name, gate_count, activity, clock_mhz, fanout))

    def estimate(self) -> Dict:
        total_dynamic = 0
        total_leakage = 0
        block_results = []

        for b in self.blocks:
            dyn = self.dynamic.block_power_mw(b)
            leak = self.leakage.block_leakage_mw(b)
            total_dynamic += dyn
            total_leakage += leak
            block_results.append({
                "name": b.name,
                "gates": b.gate_count,
                "activity": b.activity,
                "dynamic_mw": round(dyn, 3),
                "leakage_mw": round(leak, 3),
                "total_mw": round(dyn + leak, 3),
            })

        total = total_dynamic + total_leakage
        return {
            "process": self.process.name,
            "temp_c": self.temp,
            "blocks": block_results,
            "total_dynamic_mw": round(total_dynamic, 2),
            "total_leakage_mw": round(total_leakage, 2),
            "total_power_mw": round(total, 2),
            "total_power_w": round(total / 1000, 3),
            "efficiency_top_w": round(total / 1000, 3),
        }

    def estimate_vessel(self, vessel: str, n_layers: int = 24,
                        d_model: int = 512) -> Dict:
        """Quick power estimate for a vessel class."""
        configs = {
            "scout": {"mac_units": 64, "clock": 200},
            "messenger": {"mac_units": 256, "clock": 400},
            "navigator": {"mac_units": 1024, "clock": 500},
            "captain": {"mac_units": 4096, "clock": 600},
        }
        cfg = configs.get(vessel, configs["captain"])

        self.blocks = []
        # MAC array
        mac_gates = cfg["mac_units"] * 500  # ~500 gates per MAC
        self.add_block("mac_array", mac_gates, 0.3, cfg["clock"])
        # Weight banks
        wb_gates = n_layers * d_model * 2
        self.add_block("weight_banks", wb_gates, 0.05, cfg["clock"])
        # Control
        self.add_block("control", 10000, 0.2, cfg["clock"])
        # I/O
        self.add_block("io_pads", 2000, 0.1, cfg["clock"])
        # Clock tree
        self.add_block("clock_tree", 5000, 1.0, cfg["clock"])
        # SRAM
        sram_gates = n_layers * d_model * 8
        self.add_block("sram", sram_gates, 0.02, cfg["clock"])

        result = self.estimate()
        result["vessel"] = vessel
        return result


def demo():
    print("=== Chip Power Estimator ===\n")

    for vessel in ["scout", "messenger", "navigator", "captain"]:
        est = ChipPowerEstimator(28, 85)
        r = est.estimate_vessel(vessel, 24, 512)
        print(f"--- {vessel.upper()} ({r['process']}) ---")
        print(f"  Total: {r['total_power_mw']}mW ({r['total_power_w']}W)")
        print(f"  Dynamic: {r['total_dynamic_mw']}mW, Leakage: {r['total_leakage_mw']}mW")
        for b in r["blocks"]:
            pct = b["total_mw"] / r["total_power_mw"] * 100 if r["total_power_mw"] > 0 else 0
            print(f"    {b['name']:15s}: {b['total_mw']:8.3f}mW ({pct:4.1f}%)")
        print()

    # Process comparison
    print("--- Process Comparison (Captain, 24L, d=512) ---")
    for nm in [5, 7, 28, 40, 65, 130]:
        est = ChipPowerEstimator(nm, 85)
        r = est.estimate_vessel("captain", 24, 512)
        print(f"  {nm:3d}nm ({r['process']:12s}): {r['total_power_mw']:8.1f}mW "
              f"(dyn={r['total_dynamic_mw']:.1f}, leak={r['total_leakage_mw']:.1f})")


if __name__ == "__main__":
    demo()
