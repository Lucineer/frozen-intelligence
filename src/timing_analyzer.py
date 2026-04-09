#!/usr/bin/env python3
"""Static timing analysis (STA) for mask-locked inference chips.

Critical path analysis, setup/hold time checks, clock uncertainty,
and timing closure across process-voltage-temperature corners.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class PVTCorner(Enum):
    SS_0P9_125 = ("SS", 0.9, 125)   # slow-slow, low V, high T (worst setup)
    FF_1P1_N40 = ("FF", 1.1, -40)   # fast-fast, high V, low T (worst hold)
    TT_1P0_25 = ("TT", 1.0, 25)     # typical
    SS_0P9_85 = ("SS", 0.9, 85)     # slow, nominal temp
    FS_1P0_125 = ("FS", 1.0, 125)   # front-slow


@dataclass
class TimingArc:
    """Timing arc from start pin to end pin."""
    name: str
    delay_ps: float  # picoseconds
    slew_ps: float
    start_pin: str
    end_pin: str
    arc_type: str = "combinational"  # combinational, setup, hold


@dataclass
class ClockDomain:
    name: str
    period_ps: float
    uncertainty_ps: float = 50.0
    skew_ps: float = 100.0
    duty_pct: float = 50.0

    @property
    def freq_mhz(self) -> float:
        return 1e6 / self.period_ps

    @property
    def freq_ghz(self) -> float:
        return 1e3 / self.period_ps


class CellLibrary:
    """Simplified standard cell library."""

    def __init__(self, process_nm: int = 28):
        self.process_nm = process_nm
        base = process_nm / 28.0  # scaling factor
        self.cells = {
            "INV_X1": {"delay": 15 * base, "power": 0.5 * base, "area": 1},
            "INV_X4": {"delay": 8 * base, "power": 2.0 * base, "area": 4},
            "NAND2_X1": {"delay": 20 * base, "power": 0.8 * base, "area": 1.5},
            "NOR2_X1": {"delay": 25 * base, "power": 0.9 * base, "area": 1.5},
            "DFF_X1": {"delay": 35 * base, "power": 1.5 * base, "area": 4,
                       "setup": 30 * base, "hold": 5 * base},
            "BUF_X4": {"delay": 10 * base, "power": 1.5 * base, "area": 3},
            "MUX2_X1": {"delay": 30 * base, "power": 1.0 * base, "area": 3},
            "FA_X1": {"delay": 50 * base, "power": 2.0 * base, "area": 5},  # full adder
            "HA_X1": {"delay": 30 * base, "power": 1.2 * base, "area": 3},  # half adder
        }

    def pvt_scale(self, corner: PVTCorner) -> float:
        """Delay scaling factor for PVT corner."""
        process, voltage, temp = corner.value
        # Simplified scaling
        p_scale = 1.3 if process == "SS" else (0.8 if process == "FF" else
                 (1.0 if process == "TT" else 1.1))
        v_scale = (1.0 / voltage) ** 1.2  # lower V = slower
        t_scale = 1.0 + (temp - 25) * 0.003  # higher T = slower
        return p_scale * v_scale * t_scale


@dataclass
class CriticalPath:
    """Critical path report."""
    name: str
    arcs: List[TimingArc] = field(default_factory=list)
    total_delay_ps: float = 0.0
    slack_ps: float = 0.0
    met: bool = True

    def add_arc(self, arc: TimingArc):
        self.arcs.append(arc)
        self.total_delay_ps += arc.delay_ps


class StaticTimingAnalyzer:
    """STA engine."""

    def __init__(self, process_nm: int = 28, corner: PVTCorner = PVTCorner.TT_1P0_25):
        self.process_nm = process_nm
        self.corner = corner
        self.lib = CellLibrary(process_nm)
        self.scale = self.lib.pvt_scale(corner)
        self.paths: List[CriticalPath] = []
        self.violations: List[Dict] = []

    def _scaled_delay(self, cell_name: str) -> float:
        return self.lib.cells[cell_name]["delay"] * self.scale

    def analyze_mac_unit(self, n_bits: int = 8, pipeline_depth: int = 1) -> CriticalPath:
        """Analyze critical path of a single MAC unit."""
        path = CriticalPath(f"MAC_{n_bits}bit_p{pipeline_depth}")

        # Partial product generation
        pp_delay = self._scaled_delay("AND_X1") if "AND_X1" in self.lib.cells else 15 * self.scale
        path.add_arc(TimingArc("partial_product", pp_delay, pp_delay * 0.5,
                               "input", "pp_out"))

        # Wallace tree reduction
        stages = math.ceil(math.log2(n_bits * 2))
        tree_delay = stages * self._scaled_delay("FA_X1") / 2
        path.add_arc(TimingArc("wallace_tree", tree_delay, tree_delay * 0.7,
                               "pp_out", "tree_out"))

        # Final adder (carry-lookahead)
        cla_delay = math.ceil(math.log2(n_bits)) * self._scaled_delay("NAND2_X1")
        path.add_arc(TimingArc("cla_adder", cla_delay, cla_delay * 0.8,
                               "tree_out", "sum_out"))

        # Output register
        dff = self.lib.cells["DFF_X1"]
        setup = dff["setup"] * self.scale
        path.add_arc(TimingArc("output_reg", setup, setup,
                               "sum_out", "reg_out"))

        return path

    def analyze_systolic_array(self, rows: int, cols: int,
                               clock_mhz: float = 500) -> Dict:
        """Analyze timing for a systolic array."""
        clock = ClockDomain("main", period_ps=int(1e6 / clock_mhz))

        mac_path = self.analyze_mac_unit()
        mac_delay = mac_path.total_delay_ps

        # Pipelined: each MAC fits in one clock cycle
        cycles_per_mac = math.ceil(mac_delay / clock.period_ps)

        slack = clock.period_ps - mac_delay
        met = slack > 0

        # Array throughput
        throughput = clock.freq_mhz if met else clock.freq_mhz / cycles_per_mac

        return {
            "array": f"{rows}x{cols}",
            "clock_mhz": clock_mhz,
            "clock_period_ps": clock.period_ps,
            "mac_delay_ps": round(mac_delay, 1),
            "slack_ps": round(slack, 1),
            "timing_met": met,
            "cycles_per_mac": cycles_per_mac,
            "throughput_mops": round(throughput * 1e6, 0),
        }

    def analyze_layer(self, d_model: int, n_heads: int,
                      clock_mhz: float = 500) -> Dict:
        """Analyze full transformer layer timing."""
        clock = ClockDomain("main", period_ps=int(1e6 / clock_mhz))
        mac = self.analyze_mac_unit()
        mac_ps = mac.total_delay_ps

        # Attention: Q*K^T (d_model x d_model)
        attn_macs = d_model * d_model * n_heads
        attn_ps = attn_macs * mac_ps / (1 if mac_ps <= clock.period_ps else
                     math.ceil(mac_ps / clock.period_ps))

        # FFN: d_model x d_ff x 3
        d_ff = d_model * 4
        ffn_macs = d_model * d_ff * 3
        ffn_ps = ffn_macs * mac_ps / (1 if mac_ps <= clock.period_ps else
                     math.ceil(mac_ps / clock.period_ps))

        total_ps = attn_ps + ffn_ps
        slack = clock.period_ps - mac_ps
        meta = slack > 0

        return {
            "d_model": d_model, "n_heads": n_heads,
            "clock_mhz": clock_mhz,
            "mac_delay_ps": round(mac_ps, 1),
            "single_mac_met": meta,
            "attn_latency_us": round(attn_ps / 1e6, 3),
            "ffn_latency_us": round(ffn_ps / 1e6, 3),
            "total_latency_us": round(total_ps / 1e6, 3),
            "tokens_per_sec": round(1e6 / (total_ps / 1e3) * 1e3, 1) if total_ps > 0 else 0,
        }

    def check_corners(self, d_model: int, n_heads: int,
                      clock_mhz: float = 500) -> Dict:
        """Check timing across all PVT corners."""
        results = {}
        for corner in PVTCorner:
            self.corner = corner
            self.scale = self.lib.pvt_scale(corner)
            r = self.analyze_layer(d_model, n_heads, clock_mhz)
            results[corner.name] = {
                "mac_met": r["single_mac_met"],
                "tokens_per_sec": r["tokens_per_sec"],
                "mac_delay_ps": r["mac_delay_ps"],
            }
            if not r["single_mac_met"]:
                self.violations.append({
                    "corner": corner.name, "type": "setup",
                    "mac_delay_ps": r["mac_delay_ps"],
                    "period_ps": int(1e6 / clock_mhz),
                })
        return results


def demo():
    print("=== Static Timing Analysis ===\n")

    sta = StaticTimingAnalyzer(28, PVTCorner.TT_1P0_25)

    # MAC unit
    print("--- MAC Unit Critical Path ---")
    mac = sta.analyze_mac_unit(8)
    print(f"  Total delay: {mac.total_delay_ps:.1f} ps")
    print(f"  Arcs:")
    for arc in mac.arcs:
        pct = arc.delay_ps / mac.total_delay_ps * 100 if mac.total_delay_ps > 0 else 0
        print(f"    {arc.name}: {arc.delay_ps:.1f} ps ({pct:.0f}%)")
    print()

    # Systolic array
    print("--- Systolic Array Timing ---")
    for clock in [200, 400, 500, 800]:
        r = sta.analyze_systolic_array(4, 4, clock)
        status = "PASS" if r["timing_met"] else f"FAIL ({r['cycles_per_mac']} cyc/MAC)"
        print(f"  {clock}MHz: MAC={r['mac_delay_ps']:.0f}ps, slack={r['slack_ps']:.0f}ps, {status}")
    print()

    # Layer timing
    print("--- Layer Timing ---")
    for d_model in [256, 512, 1024]:
        r = sta.analyze_layer(d_model, 8, 500)
        print(f"  d={d_model}: {r['attn_latency_us']:.1f}us attn, "
              f"{r['ffn_latency_us']:.1f}us ffn, {r['tokens_per_sec']:.0f} tok/s")
    print()

    # PVT corners
    print("--- PVT Corner Analysis (d=512, 500MHz) ---")
    sta2 = StaticTimingAnalyzer(28, PVTCorner.TT_1P0_25)
    corners = sta2.check_corners(512, 8, 500)
    for name, r in corners.items():
        status = "PASS" if r["mac_met"] else "FAIL"
        print(f"  {name}: MAC={r['mac_delay_ps']:.0f}ps, {status}")


if __name__ == "__main__":
    demo()
