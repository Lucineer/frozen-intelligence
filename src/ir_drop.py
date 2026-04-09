#!/usr/bin/env python3
"""IR drop analyzer for mask-locked chip power delivery networks.

Models VDD/GND rail resistance, current density, and voltage
drop across the power grid during peak activity.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class PowerGridNode:
    x: int
    y: int
    vdd_voltage: float
    resistance_to_parent: float  # ohms (VDD rail segment)
    current_draw: float          # amps
    is_pad: bool = False


class IRDropAnalyzer:
    """Analyze voltage drop across chip power grid."""

    def __init__(self, grid_size: int = 10, die_size_mm: float = 5.0,
                 vdd_nominal: float = 0.9, process_nm: int = 28):
        self.n = grid_size
        self.die_size = die_size_mm
        self.vdd = vdd_nominal
        self.nm = process_nm
        self.cell_size = die_size_mm / grid_size  # mm

        # Metal layer sheet resistances (ohm/sq) by process
        resistivities = {
            28: {"M1": 0.3, "M2": 0.2, "M3": 0.15, "M4": 0.1, "M5": 0.05, "M6": 0.03},
            40: {"M1": 0.4, "M2": 0.3, "M3": 0.2, "M4": 0.12, "M5": 0.06},
            65: {"M1": 0.6, "M2": 0.4, "M3": 0.3, "M4": 0.15},
            130: {"M1": 0.8, "M2": 0.6, "M3": 0.4},
        }
        self.sheet_rho = resistivities.get(process_nm, resistivities[28])

        # Power grid nodes
        self.nodes: List[List[PowerGridNode]] = []
        self.pad_spacing = 2
        self._build_grid()

    def _build_grid(self):
        self.nodes = []
        for j in range(self.n):
            row = []
            for i in range(self.n):
                # Rail resistance: 100um * sheet_rho / (wire_width/wire_pitch)
                rail_width_um = 5.0  # 5um power rail
                seg_len_um = self.cell_size * 1000  # mm -> um
                r = self.sheet_rho["M1"] * (seg_len_um / rail_width_um)
                is_pad = (i % self.pad_spacing == 0 and j % self.pad_spacing == 0)
                node = PowerGridNode(i, j, self.vdd, r, 0.0, is_pad)
                row.append(node)
            self.nodes.append(row)

    def add_power_source(self, x_mm: float, y_mm: float, current_a: float):
        gx = min(int(x_mm / self.cell_size), self.n - 1)
        gy = min(int(y_mm / self.cell_size), self.n - 1)
        self.nodes[gy][gx].is_pad = True
        self.nodes[gy][gx].current_draw = -current_a  # negative = supply

    def add_load(self, x_mm: float, y_mm: float, w_mm: float, h_mm: float,
                 power_mw: float):
        gx = min(int(x_mm / self.cell_size), self.n - 1)
        gy = min(int(y_mm / self.cell_size), self.n - 1)
        current = (power_mw * 1e-3) / self.vdd  # I = P/V
        cells_w = max(1, int(w_mm / self.cell_size))
        cells_h = max(1, int(h_mm / self.cell_size))
        per_cell = current / (cells_w * cells_h)
        for dj in range(cells_h):
            for di in range(cells_w):
                nj, ni = gy + dj, gx + di
                if 0 <= nj < self.n and 0 <= ni < self.n:
                    self.nodes[nj][ni].current_draw += per_cell

    def analyze(self) -> Dict:
        """Compute static IR drop across power grid."""
        # Iterative Gauss-Seidel for voltage
        for iteration in range(100):
            max_delta = 0
            for j in range(self.n):
                for i in range(self.n):
                    node = self.nodes[j][i]
                    if node.is_pad:
                        node.vdd_voltage = self.vdd
                        continue

                    neighbors = []
                    if i > 0: neighbors.append(self.nodes[j][i-1])
                    if i < self.n-1: neighbors.append(self.nodes[j][i+1])
                    if j > 0: neighbors.append(self.nodes[j-1][i])
                    if j < self.n-1: neighbors.append(self.nodes[j+1][i])

                    g_sum = sum(1.0 / (n.resistance_to_parent + 1e-12) for n in neighbors)
                    if g_sum > 0:
                        v_new = sum(n.vdd_voltage / (n.resistance_to_parent + 1e-12) for n in neighbors)
                        v_new = v_new / g_sum - node.current_draw / g_sum
                        v_new = max(0, min(self.vdd, v_new))
                        delta = abs(v_new - node.vdd_voltage)
                        max_delta = max(max_delta, delta)
                        node.vdd_voltage = v_new

            if max_delta < 1e-6:
                break

        # Collect results
        voltages = [self.nodes[j][i].vdd_voltage
                    for j in range(self.n) for i in range(self.n)]
        min_v = min(voltages)
        max_drop = (self.vdd - min_v) * 1000  # mV

        # Worst IR drop location
        min_pos = (0, 0)
        for j in range(self.n):
            for i in range(self.n):
                if self.nodes[j][i].vdd_voltage < self.nodes[min_pos[1]][min_pos[0]].vdd_voltage:
                    min_pos = (i, j)

        return {
            "vdd_nominal": self.vdd,
            "min_voltage": round(min_v, 4),
            "worst_ir_drop_mv": round(max_drop, 1),
            "max_allowed_drop_pct": 5.0,
            "passes": max_drop < self.vdd * 50,  # 5% of VDD
            "worst_location_mm": (round(min_pos[0] * self.cell_size, 2),
                                  round(min_pos[1] * self.cell_size, 2)),
            "iterations": iteration + 1,
        }

    def voltage_map_str(self) -> str:
        lines = []
        for j in range(self.n):
            row = ""
            for i in range(self.n):
                v = self.nodes[j][i].vdd_voltage
                drop_pct = (self.vdd - v) / self.vdd * 100
                if self.nodes[j][i].is_pad:
                    row += "P"
                elif drop_pct < 1:
                    row += "."
                elif drop_pct < 2:
                    row += "+"
                elif drop_pct < 3:
                    row += "="
                elif drop_pct < 5:
                    row += "#"
                else:
                    row += "!"
            lines.append(row)
        return "\n".join(lines)


def demo():
    print("=== IR Drop Analyzer ===\n")

    ir = IRDropAnalyzer(10, 5.0, 0.9, 28)
    print(f"Grid: {ir.n}x{ir.n}, Die: {ir.die_size}mm, VDD: {ir.vdd}V")
    print(f"Sheet rho (M1): {ir.sheet_rho['M1']} ohm/sq, Pad spacing: every {ir.pad_spacing} cells")
    print()

    # Central power load (MAC array)
    ir.add_load(1.5, 1.5, 2.0, 2.0, 2000)   # 2W MAC array
    ir.add_load(0.5, 0.5, 0.5, 0.5, 300)     # 300mW SRAM
    ir.add_load(4.0, 4.0, 0.8, 0.8, 100)     # 100mW IO

    print("--- Power Grid Layout ---")
    total_current = sum(ir.nodes[j][i].current_draw
                       for j in range(ir.n) for i in range(ir.n))
    print(f"  Total current draw: {abs(total_current):.2f}A")
    print(f"  Pads: {sum(1 for j in range(ir.n) for i in range(ir.n) if ir.nodes[j][i].is_pad)}")
    print()

    result = ir.analyze()
    print("--- IR Drop Results ---")
    print(f"  VDD nominal: {result['vdd_nominal']}V")
    print(f"  Min voltage: {result['min_voltage']}V")
    print(f"  Worst IR drop: {result['worst_ir_drop_mv']}mV")
    print(f"  Worst location: {result['worst_location_mm']}mm")
    print(f"  Passes (<5% VDD): {result['passes']}")
    print(f"  Converged in: {result['iterations']} iterations")
    print()

    print("--- Voltage Map (P=pad, .=<1%, +=2%, ==3%, #=5%, !=fail) ---")
    print(ir.voltage_map_str())
    print()

    # Compare processes
    print("--- Process Comparison (2W load, 10x10 grid) ---")
    for nm in [28, 40, 65, 130]:
        ir2 = IRDropAnalyzer(10, 5.0, 0.9, nm)
        ir2.add_load(1.5, 1.5, 2.0, 2.0, 2000)
        r = ir2.analyze()
        print(f"  {nm}nm: IR drop = {r['worst_ir_drop_mv']}mV, "
              f"min V = {r['min_voltage']}V, pass={r['passes']}")

    # Compare with more pads
    print("\n--- Pad Density Impact (28nm, 2W) ---")
    for spacing in [1, 2, 3, 5]:
        ir3 = IRDropAnalyzer(10, 5.0, 0.9, 28)
        ir3.pad_spacing = spacing
        ir3.add_load(1.5, 1.5, 2.0, 2.0, 2000)
        r = ir3.analyze()
        pads = sum(1 for j in range(ir3.n) for i in range(ir3.n) if ir3.nodes[j][i].is_pad)
        print(f"  Spacing={spacing}: {pads:2d} pads, "
              f"IR drop={r['worst_ir_drop_mv']:.1f}mV, pass={r['passes']}")


if __name__ == "__main__":
    demo()
