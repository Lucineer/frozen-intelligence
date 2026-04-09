#!/usr/bin/env python3
"""RTL synthesis area estimator for mask-locked chips.

Estimates gate count, die area, and utilization from RTL descriptions
using technology-specific cell library data.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class CellCategory(Enum):
    COMBINATIONAL = "comb"
    SEQUENTIAL = "seq"
    MEMORY = "mem"
    ANALOG = "analog"


@dataclass
class StdCell:
    name: str
    category: CellCategory
    area_um2: float
    delay_ps: float = 0
    power_nw: float = 0
    drive_strength: int = 1


class TechLib28nm:
    """TSMC 28nm LP standard cell library (simplified)."""

    CELLS = {
        "INV_X1": StdCell("INV_X1", CellCategory.COMBINATIONAL, 0.99, 3.5, 0.5),
        "INV_X2": StdCell("INV_X2", CellCategory.COMBINATIONAL, 1.40, 2.5, 0.8),
        "INV_X4": StdCell("INV_X4", CellCategory.COMBINATIONAL, 2.10, 1.8, 1.2),
        "INV_X8": StdCell("INV_X8", CellCategory.COMBINATIONAL, 3.36, 1.3, 2.0),
        "NAND2_X1": StdCell("NAND2_X1", CellCategory.COMBINATIONAL, 1.26, 5.0, 0.6),
        "NAND2_X2": StdCell("NAND2_X2", CellCategory.COMBINATIONAL, 1.75, 3.5, 0.9),
        "NAND2_X4": StdCell("NAND2_X4", CellCategory.COMBINATIONAL, 2.66, 2.5, 1.4),
        "NOR2_X1": StdCell("NOR2_X1", CellCategory.COMBINATIONAL, 1.44, 5.5, 0.7),
        "NOR2_X2": StdCell("NOR2_X2", CellCategory.COMBINATIONAL, 2.00, 4.0, 1.0),
        "AND2_X1": StdCell("AND2_X1", CellCategory.COMBINATIONAL, 1.80, 6.0, 0.8),
        "OR2_X1": StdCell("OR2_X1", CellCategory.COMBINATIONAL, 1.80, 6.0, 0.8),
        "XOR2_X1": StdCell("XOR2_X1", CellCategory.COMBINATIONAL, 2.50, 8.0, 1.0),
        "XOR2_X2": StdCell("XOR2_X2", CellCategory.COMBINATIONAL, 3.50, 5.5, 1.5),
        "MUX2_X1": StdCell("MUX2_X1", CellCategory.COMBINATIONAL, 2.40, 7.0, 1.2),
        "MUX2_X2": StdCell("MUX2_X2", CellCategory.COMBINATIONAL, 3.20, 5.0, 1.6),
        "FA_X1": StdCell("FA_X1", CellCategory.COMBINATIONAL, 5.50, 12.0, 2.5),
        "HA_X1": StdCell("HA_X1", CellCategory.COMBINATIONAL, 3.80, 8.0, 1.8),
        "DFF_X1": StdCell("DFF_X1", CellCategory.SEQUENTIAL, 4.30, 45.0, 3.0),
        "DFF_X2": StdCell("DFF_X2", CellCategory.SEQUENTIAL, 5.60, 40.0, 4.5),
        "DFF_Q_X1": StdCell("DFF_Q_X1", CellCategory.SEQUENTIAL, 5.50, 50.0, 3.5),
        "SDFF_X1": StdCell("SDFF_X1", CellCategory.SEQUENTIAL, 5.00, 48.0, 3.2),
        "BUFG_X8": StdCell("BUFG_X8", CellCategory.COMBINATIONAL, 12.0, 30.0, 8.0),
        "RAM64X1D": StdCell("RAM64X1D", CellCategory.MEMORY, 25.0, 800.0, 50.0),
        "RAM128X1D": StdCell("RAM128X1D", CellCategory.MEMORY, 42.0, 1000.0, 80.0),
        "FIFO18X1": StdCell("FIFO18X1", CellCategory.MEMORY, 80.0, 1500.0, 120.0),
        "PLL": StdCell("PLL", CellCategory.ANALOG, 5000.0, 0, 5000.0),
        "IO_PAD": StdCell("IO_PAD", CellCategory.ANALOG, 800.0, 0, 200.0),
        "LVDS_TX": StdCell("LVDS_TX", CellCategory.ANALOG, 1200.0, 0, 800.0),
        "LVDS_RX": StdCell("LVDS_RX", CellCategory.ANALOG, 1200.0, 0, 600.0),
    }

    NAND2_EQUIV_AREA = 1.26  # 1 NAND2 = 1 equivalent gate


@dataclass
class ModuleEstimate:
    name: str
    cell_counts: Dict[str, int] = field(default_factory=dict)
    total_area_um2: float = 0
    equiv_gates: int = 0
    total_power_mw: float = 0


class AreaEstimator:
    """Estimate area from cell-level description."""

    def __init__(self, process_nm: int = 28):
        self.process_nm = process_nm
        self.lib = TechLib28nm.CELLS if process_nm == 28 else TechLib28nm.CELLS
        self.scale = (process_nm / 28.0) ** 2  # area scales with feature size^2

    def estimate_module(self, name: str,
                        cell_counts: Dict[str, int]) -> ModuleEstimate:
        total_area = 0
        total_power = 0
        for cell_name, count in cell_counts.items():
            cell = self.lib.get(cell_name)
            if cell:
                total_area += cell.area_um2 * count * self.scale
                total_power += cell.power_nw * count * self.scale
            else:
                total_area += 1.26 * count * self.scale  # default NAND2 equiv

        equiv_gates = int(total_area / (TechLib28nm.NAND2_EQUIV_AREA * self.scale))

        return ModuleEstimate(name, cell_counts, total_area, equiv_gates, total_power / 1000)

    def estimate_multiplier(self, n_bits: int) -> ModuleEstimate:
        """Estimate area for N-bit array multiplier."""
        counts = {}
        # Partial product AND gates
        counts["AND2_X1"] = n_bits * n_bits
        # Full adders for Wallace tree reduction
        n_pp = n_bits * n_bits
        n_ha = n_bits - 1  # first row uses half adders
        counts["HA_X1"] = n_ha
        counts["FA_X1"] = int(n_pp * 1.5)  # wallace tree uses ~1.5 FA per bit of result
        # Final adder (carry-lookahead)
        cla_nand = n_bits * math.ceil(math.log2(n_bits))
        counts["NAND2_X4"] = cla_nand
        counts["INV_X2"] = cla_nand // 2
        return self.estimate_module(f"multiplier_{n_bits}b", counts)

    def estimate_mac_unit(self, n_bits: int) -> ModuleEstimate:
        """MAC = multiplier + accumulator register."""
        counts = {"AND2_X1": n_bits * n_bits, "HA_X1": n_bits - 1,
                  "FA_X1": int(n_bits * n_bits * 1.5) + n_bits * 2,
                  "NAND2_X4": n_bits * math.ceil(math.log2(n_bits)),
                  "INV_X2": n_bits * math.ceil(math.log2(n_bits)) // 2,
                  "DFF_X1": n_bits * 2}
        return self.estimate_module(f"mac_{n_bits}b", counts)

    def estimate_systolic_array(self, rows: int, cols: int,
                                n_bits: int = 8) -> ModuleEstimate:
        mac = self.estimate_mac_unit(n_bits)
        total_counts = {k: v * rows * cols for k, v in mac.cell_counts.items()}
        return self.estimate_module(f"systolic_{rows}x{cols}", total_counts)

    def estimate_weight_bank(self, n_entries: int, data_width: int = 8) -> ModuleEstimate:
        """Estimate weight bank (ROM-style)."""
        counts = {}
        # Use 64x1 RAM cells
        n_cells = math.ceil(n_entries * data_width / 64)
        counts["RAM64X1D"] = n_cells
        # Address decoder
        addr_bits = math.ceil(math.log2(n_entries)) if n_entries > 1 else 1
        counts["INV_X2"] = addr_bits * 2
        counts["NAND2_X2"] = addr_bits * 2
        # Output buffer
        counts["DFF_X1"] = data_width
        return self.estimate_module(f"weight_bank_{n_entries}x{data_width}", counts)

    def estimate_full_chip(self, n_layers: int = 24, n_heads: int = 8,
                           d_model: int = 512) -> Dict:
        """Full chip area estimation."""
        results = {}

        # MAC array
        mac = self.estimate_systolic_array(n_heads, n_heads, 8)
        results["mac_array"] = mac

        # Weight banks (one per layer)
        wb = self.estimate_weight_bank(d_model, 8)
        results["weight_banks"] = ModuleEstimate(
            f"all_weight_banks ({n_layers} layers)",
            {k: v * n_layers for k, v in wb.cell_counts.items()},
            wb.total_area_um2 * n_layers,
            wb.equiv_gates * n_layers,
            wb.total_power_mw * n_layers)

        # Control logic
        ctrl_counts = {"DFF_X1": 500, "NAND2_X2": 2000, "INV_X2": 1000,
                       "MUX2_X1": 500, "DFF_X2": 200}
        results["control"] = self.estimate_module("control_fsm", ctrl_counts)

        # Clock tree
        clk_counts = {"BUFG_X8": n_heads * 4, "INV_X4": 500, "DFF_X1": 200}
        results["clock_tree"] = self.estimate_module("clock_tree", clk_counts)

        # PLL
        results["pll"] = self.estimate_module("pll", {"PLL": 1})

        # I/O pads
        results["io"] = self.estimate_module("io_ring", {"IO_PAD": 40,
                                                          "LVDS_TX": 4, "LVDS_RX": 4})

        # SRAM
        sram_counts = {"RAM128X1D": n_layers * d_model * 4}
        results["sram"] = self.estimate_module("on_chip_sram", sram_counts)

        # Totals
        total_area = sum(r.total_area_um2 for r in results.values())
        total_gates = sum(r.equiv_gates for r in results.values())
        total_power = sum(r.total_power_mw for r in results.values())

        return {
            "modules": {k: {"area_um2": round(v.total_area_um2, 1),
                           "equiv_gates": v.equiv_gates,
                           "power_mw": round(v.total_power_mw, 2)}
                       for k, v in results.items()},
            "total_area_um2": round(total_area, 1),
            "total_area_mm2": round(total_area / 1e6, 4),
            "total_equiv_gates": total_gates,
            "total_power_mw": round(total_power, 2),
            "process_nm": self.process_nm,
        }


def demo():
    print("=== RTL Synthesis Area Estimator ===\n")

    est = AreaEstimator(28)

    # Basic cells
    print("--- Cell Library (28nm) ---")
    for name, cell in sorted(TechLib28nm.CELLS.items(),
                             key=lambda x: x[1].area_um2)[:8]:
        print(f"  {name:12s}: {cell.area_um2:7.2f} um2, {cell.delay_ps:5.1f} ps, {cell.power_nw:.1f} nW")
    print(f"  ... ({len(TechLib28nm.CELLS)} total cells)")
    print()

    # Multipliers
    print("--- Multiplier Area ---")
    for bits in [4, 8, 16, 32]:
        m = est.estimate_multiplier(bits)
        print(f"  {bits:2d}-bit: {m.equiv_gates:7d} equiv gates, {m.total_area_um2:10.1f} um2")
    print()

    # Systolic arrays
    print("--- Systolic Array Area ---")
    for size in [4, 8, 16, 32]:
        a = est.estimate_systolic_array(size, size, 8)
        print(f"  {size}x{size}: {a.equiv_gates:10d} equiv gates, "
              f"{a.total_area_um2:12.0f} um2, {a.total_power_mw:.1f} mW")
    print()

    # Full chip
    print("--- Full Chip Estimate (28nm, 24L, 8H, d=512) ---")
    chip = est.estimate_full_chip()
    for name, m in chip["modules"].items():
        pct = m["area_um2"] / chip["total_area_um2"] * 100
        print(f"  {name:18s}: {m['area_um2']:12.0f} um2 ({pct:5.1f}%), "
              f"{m['equiv_gates']:8d} gates, {m['power_mw']:.1f} mW")
    print(f"  {'TOTAL':18s}: {chip['total_area_um2']:12.0f} um2, "
          f"{chip['total_equiv_gates']:8d} gates, {chip['total_power_mw']:.1f} mW")
    die_side = math.sqrt(chip['total_area_um2'] / 0.5)  # 50% utilization
    print(f"  Die size (50% util): {die_side:.0f}um x {die_side:.0f}um")
    print()

    # Process scaling
    print("--- Process Scaling ---")
    for nm in [28, 40, 65, 130]:
        e = AreaEstimator(nm)
        m = e.estimate_multiplier(8)
        print(f"  {nm:3d}nm: 8-bit mult = {m.total_area_um2:.1f} um2 "
              f"({m.total_area_um2 / e.estimate_multiplier(8).total_area_um2:.1f}x vs self)")


if __name__ == "__main__":
    demo()
