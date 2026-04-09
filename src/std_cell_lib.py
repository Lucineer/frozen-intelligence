#!/usr/bin/env python3
"""Standard cell library for mask-locked chip synthesis.

Cell characterization, drive strength selection, timing arcs,
and power estimates for synthesis optimization.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class CellPin:
    name: str
    direction: str  # input, output
    capacitance_ff: float = 0
    max_fanout: int = 4


@dataclass
class TimingArc:
    from_pin: str
    to_pin: str
    delay_rise_ps: float
    delay_fall_ps: float
    slew_rise_ps: float
    slew_fall_ps: float


@dataclass
class StdCell:
    name: str
    function: str       # INV, NAND2, NOR2, BUF, etc
    drive_strength: int  # 1x, 2x, 4x, 8x
    width_um: float
    height_um: float
    area_um2: float
    leakage_nw: float    # nW
    pins: List[CellPin] = field(default_factory=list)
    timing_arcs: List[TimingArc] = field(default_factory=list)

    @property
    def gate_cap_ff(self) -> float:
        for p in self.pins:
            if p.direction == "input":
                return p.capacitance_ff
        return 0


class CellLibrary:
    """Standard cell library with characterization."""

    def __init__(self, process_nm: int = 28):
        self.nm = process_nm
        self.cells: Dict[str, StdCell] = {}
        self.cell_height = 1.2 if process_nm <= 40 else 1.6  # um

        # Base delays per unit drive strength (ps) at 28nm
        self.base_delays = {
            "INV":  {"rise": 8, "fall": 10, "slew": 12, "width": 0.4},
            "NAND2": {"rise": 10, "fall": 12, "slew": 15, "width": 0.8},
            "NOR2": {"rise": 12, "fall": 10, "slew": 15, "width": 0.8},
            "NAND3": {"rise": 12, "fall": 15, "slew": 18, "width": 1.2},
            "NOR3": {"rise": 15, "fall": 12, "slew": 18, "width": 1.2},
            "BUF":  {"rise": 5, "fall": 6, "slew": 8, "width": 0.8},
            "AOI21": {"rise": 14, "fall": 12, "slew": 18, "width": 1.0},
            "OAI21": {"rise": 12, "fall": 14, "slew": 18, "width": 1.0},
            "XOR2": {"rise": 18, "fall": 18, "slew": 22, "width": 1.2},
            "XNOR2": {"rise": 18, "fall": 18, "slew": 22, "width": 1.2},
            "MUX2": {"rise": 16, "fall": 16, "slew": 20, "width": 1.4},
            "DFF":  {"rise": 30, "fall": 25, "slew": 25, "width": 2.0},
            "ADD":  {"rise": 40, "fall": 40, "slew": 45, "width": 3.0},
            "FA":   {"rise": 35, "fall": 35, "slew": 40, "width": 2.5},
        }

        # Process scaling factor
        scale = {28: 1.0, 40: 1.4, 65: 2.2, 130: 4.5}
        self.scale_factor = scale.get(process_nm, 1.0)

        self._build_library()

    def _build_library(self):
        for func, params in self.base_delays.items():
            for strength in [1, 2, 4, 8]:
                sf = self.scale_factor
                w = params["width"] * strength * sf
                # Delay decreases with drive strength (not perfectly linear)
                delay_factor = 1.0 / math.sqrt(strength)

                name = f"{func}_X{strength}"
                rise = params["rise"] * delay_factor * sf
                fall = params["fall"] * delay_factor * sf
                slew = params["slew"] * delay_factor * sf

                # Build pins
                pins = []
                if func in ("INV", "BUF"):
                    pins.append(CellPin("A", "input", round(0.5 * strength, 1)))
                    pins.append(CellPin("Y", "output", round(0.3 * strength, 1)))
                elif func == "DFF":
                    pins.append(CellPin("D", "input", 1.0))
                    pins.append(CellPin("CK", "input", 1.0))
                    pins.append(CellPin("Q", "output", 1.0))
                elif func in ("ADD", "FA"):
                    pins.append(CellPin("A", "input", 0.8))
                    pins.append(CellPin("B", "input", 0.8))
                    pins.append(CellPin("CI", "input", 0.8))
                    pins.append(CellPin("CO", "output", 0.8))
                    pins.append(CellPin("S", "output", 0.8))
                elif func == "MUX2":
                    pins.append(CellPin("D0", "input", 0.8))
                    pins.append(CellPin("D1", "input", 0.8))
                    pins.append(CellPin("S", "input", 0.5))
                    pins.append(CellPin("Y", "output", 0.8))
                else:
                    pins.append(CellPin("A", "input", round(0.5 * strength, 1)))
                    pins.append(CellPin("B", "input", round(0.5 * strength, 1)))
                    if "3" in func:
                        pins.append(CellPin("C", "input", round(0.5 * strength, 1)))
                    pins.append(CellPin("Y", "output", round(0.3 * strength, 1)))

                # Timing arc
                arcs = []
                for p in pins:
                    if p.direction == "input":
                        arcs.append(TimingArc(p.name, "Y", rise, fall, slew, slew))

                leakage = w * 0.5  # 0.5 nW/um
                cell = StdCell(
                    name=name, function=func, drive_strength=strength,
                    width_um=round(w, 2), height_um=self.cell_height,
                    area_um2=round(w * self.cell_height, 2),
                    leakage_nw=round(leakage, 2),
                    pins=pins, timing_arcs=arcs,
                )
                self.cells[name] = cell

    def get_cell(self, name: str) -> Optional[StdCell]:
        return self.cells.get(name)

    def select_drive_strength(self, func: str, target_delay_ps: float) -> str:
        """Select smallest drive strength meeting timing."""
        for strength in [1, 2, 4, 8]:
            name = f"{func}_X{strength}"
            cell = self.cells.get(name)
            if cell and cell.timing_arcs:
                max_delay = max(cell.timing_arcs[0].delay_rise_ps,
                              cell.timing_arcs[0].delay_fall_ps)
                if max_delay <= target_delay_ps:
                    return name
        return f"{func}_X8"  # max available

    def area_estimate(self, instance_counts: Dict[str, int]) -> Dict:
        total_area = 0
        total_leakage = 0
        for name, count in instance_counts.items():
            cell = self.cells.get(name)
            if cell:
                total_area += cell.area_um2 * count
                total_leakage += cell.leakage_nw * count
        return {"total_area_um2": round(total_area, 1),
                "total_leakage_uw": round(total_leakage / 1000, 3),
                "utilization_pct": round(total_area / (2500 * 2500) * 100, 1)}


def demo():
    print("=== Standard Cell Library ===\n")

    lib = CellLibrary(28)
    print(f"Process: {lib.nm}nm, Cell height: {lib.cell_height}um")
    print(f"Total cells: {len(lib.cells)}")
    print()

    # Show INV variants
    print("--- INV Drive Strengths ---")
    for s in [1, 2, 4, 8]:
        c = lib.cells[f"INV_X{s}"]
        arc = c.timing_arcs[0]
        print(f"  INV_X{s:1d}: {c.width_um:.2f}um wide, "
              f"delay={arc.delay_rise_ps:.0f}/{arc.delay_fall_ps:.0f}ps, "
              f"slew={arc.slew_rise_ps:.0f}ps, "
              f"area={c.area_um2:.1f}um2, "
              f"leak={c.leakage_nw:.1f}nW")
    print()

    # Drive strength selection
    print("--- Drive Strength Selection ---")
    for target in [20, 10, 5, 3]:
        sel = lib.select_drive_strength("INV", target)
        c = lib.cells[sel]
        print(f"  Target {target}ps -> {sel} ({c.width_um:.2f}um)")

    print()

    # Show all cell functions
    print("--- All Cell Functions ---")
    funcs = {}
    for name, cell in lib.cells.items():
        if cell.function not in funcs:
            funcs[cell.function] = []
        funcs[cell.function].append(name)

    for func in sorted(funcs.keys()):
        variants = ", ".join(funcs[func])
        print(f"  {func:8s}: {variants}")

    print()

    # Area estimate for a simple design
    print("--- Area Estimate (256-bit adder = 256 FA) ---")
    counts = {"FA_X4": 256, "BUF_X2": 16, "INV_X1": 32, "DFF_X2": 256}
    est = lib.area_estimate(counts)
    print(f"  Total area: {est['total_area_um2']:.0f} um2 ({est['total_area_um2']/1e6:.2f} mm2)")
    print(f"  Total leakage: {est['total_leakage_uw']:.3f} uW")
    print(f"  Die utilization (5mm die): {est['utilization_pct']}%")

    # Process comparison
    print("\n--- Process Comparison (INV_X1) ---")
    for nm in [28, 40, 65, 130]:
        l = CellLibrary(nm)
        c = l.cells["INV_X1"]
        arc = c.timing_arcs[0]
        print(f"  {nm}nm: width={c.width_um:.2f}um, "
              f"delay={arc.delay_rise_ps:.0f}/{arc.delay_fall_ps:.0f}ps, "
              f"area={c.area_um2:.1f}um2")


if __name__ == "__main__":
    demo()
