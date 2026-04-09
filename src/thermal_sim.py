#!/usr/bin/env python3
"""2D thermal simulator for mask-locked inference chips.

Finite-difference heat diffusion across die, with power sources,
thermal vias, and package thermal resistance.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ThermalSource:
    name: str
    x: int  # grid position
    y: int
    width: int  # grid cells
    height: int
    power_mw: float  # mW


class ThermalGrid:
    """2D finite-difference thermal grid."""

    def __init__(self, grid_size: int = 50, die_size_mm: float = 5.0):
        self.n = grid_size
        self.die_size = die_size_mm
        self.dx = die_size_mm / grid_size  # mm per cell

        # Material properties (silicon)
        self.k_si = 148.0  # W/(m*K) thermal conductivity
        self.k_cu = 400.0  # copper thermal via
        self.thickness_mm = 0.3  # die thickness

        # Temperature grid (Celsius)
        self.T = [[25.0] * grid_size for _ in range(grid_size)]
        self.T_ambient = 45.0  # package ambient

        # Power density grid (W/m2)
        self.power = [[0.0] * grid_size for _ in range(grid_size)]

        # Thermal via enhancement map
        self.via_map = [[1.0] * grid_size for _ in range(grid_size)]

        self.sources: List[ThermalSource] = []
        self.time_step = 1e-6  # seconds
        self.alpha = 0.0  # thermal diffusivity, computed from k

        self._update_alpha()

    def _update_alpha(self):
        rho = 2329.0  # silicon density kg/m3
        cp = 700.0    # specific heat J/(kg*K)
        self.alpha = self.k_si / (rho * cp)

    def add_source(self, name: str, x_mm: float, y_mm: float,
                   w_mm: float, h_mm: float, power_mw: float):
        gx = int(x_mm / self.dx)
        gy = int(y_mm / self.dx)
        gw = max(1, int(w_mm / self.dx))
        gh = max(1, int(h_mm / self.dx))
        self.sources.append(ThermalSource(name, gx, gy, gw, gh, power_mw))

        # Fill power grid
        power_density = (power_mw * 1e-3) / (w_mm * h_mm * 1e-6)  # W/m2
        for yi in range(gy, min(gy + gh, self.n)):
            for xi in range(gx, min(gx + gw, self.n)):
                self.power[yi][xi] += power_density

    def add_thermal_vias(self, x_mm: float, y_mm: float,
                         w_mm: float, h_mm: float):
        gx = int(x_mm / self.dx)
        gy = int(y_mm / self.dx)
        gw = max(1, int(w_mm / self.dx))
        gh = max(1, int(h_mm / self.dx))
        enhancement = self.k_cu / self.k_si
        for yi in range(gy, min(gy + gh, self.n)):
            for xi in range(gx, min(gx + gw, self.n)):
                self.via_map[yi][xi] = enhancement

    def step(self, n_steps: int = 1) -> Dict:
        """Run finite-difference simulation. Returns stats."""
        dx_m = self.dx * 1e-3
        dt = 0.25 * dx_m * dx_m / max(self.alpha, 1e-20)  # stability
        dt = min(dt, self.time_step)

        for _ in range(n_steps):
            new_T = [[0.0] * self.n for _ in range(self.n)]
            for j in range(self.n):
                for i in range(self.n):
                    k = self.k_si * self.via_map[j][i]
                    alpha_eff = k / (2329.0 * 700.0)

                    # Laplacian with boundary conditions
                    T_up = self.T[j - 1][i] if j > 0 else self.T_ambient
                    T_down = self.T[j + 1][i] if j < self.n - 1 else self.T_ambient
                    T_left = self.T[j][i - 1] if i > 0 else self.T[j][i]
                    T_right = self.T[j][i + 1] if i < self.n - 1 else self.T[j][i]

                    laplacian = (T_up + T_down + T_left + T_right - 4 * self.T[j][i]) / (dx_m * dx_m)
                    power_term = self.power[j][i] / (2329.0 * 700.0 * self.thickness_mm * 1e-3)

                    new_T[j][i] = self.T[j][i] + dt * (alpha_eff * laplacian + power_term)

            self.T = new_T

        return self.get_stats()

    def get_stats(self) -> Dict:
        flat = [self.T[j][i] for j in range(self.n) for i in range(self.n)]
        max_t = max(flat)
        min_t = min(flat)
        avg_t = sum(flat) / len(flat)
        return {"max_C": round(max_t, 1), "min_C": round(min_t, 1),
                "avg_C": round(avg_t, 1), "delta_C": round(max_t - min_t, 1)}

    def get_hotspot(self) -> Dict:
        max_t = 25
        max_pos = (0, 0)
        for j in range(self.n):
            for i in range(self.n):
                if self.T[j][i] > max_t:
                    max_t = self.T[j][i]
                    max_pos = (i, j)
        return {"x_mm": round(max_pos[0] * self.dx, 3),
                "y_mm": round(max_pos[1] * self.dx, 3),
                "temp_C": round(max_t, 1)}

    def thermal_map_str(self, resolution: int = 20) -> str:
        """ASCII thermal map."""
        step = max(1, self.n // resolution)
        lines = []
        for j in range(0, self.n, step):
            row = ""
            for i in range(0, self.n, step):
                t = self.T[j][i]
                if t < 30:
                    row += " "
                elif t < 50:
                    row += "."
                elif t < 60:
                    row += "+"
                elif t < 70:
                    row += "="
                elif t < 80:
                    row += "*"
                elif t < 90:
                    row += "#"
                else:
                    row += "@"
            lines.append(row)
        return "\n".join(lines)


def demo():
    print("=== 2D Thermal Simulator ===\n")

    grid = ThermalGrid(50, 5.0)
    print(f"Grid: {grid.n}x{grid.n}, Die: {grid.die_size}mm, "
          f"Cell: {grid.dx*1000:.0f}um")
    print(f"Ambient: {grid.T_ambient}C, k_Si: {grid.k_si} W/(m*K)")
    print()

    # Add power sources (typical chip layout)
    grid.add_source("MAC array", 1.0, 1.0, 3.0, 3.0, 2000)    # 2W central
    grid.add_source("SRAM", 0.5, 0.5, 1.0, 1.0, 500)           # 0.5W corner
    grid.add_source("IO pads", 0.0, 0.0, 5.0, 0.2, 100)       # 0.1W edge
    grid.add_source("IO pads", 0.0, 4.8, 5.0, 0.2, 100)       # bottom edge

    # Add thermal vias under MAC array
    grid.add_thermal_vias(1.5, 1.5, 2.0, 2.0)

    print(f"Power sources: {len(grid.sources)}")
    for s in grid.sources:
        print(f"  {s.name}: ({s.x},{s.y}) {s.width}x{s.height} = {s.power_mw}mW")
    print()

    # Run simulation
    print("--- Steady-State (1000 steps) ---")
    stats = grid.step(1000)
    print(f"  Max: {stats['max_C']}C, Min: {stats['min_C']}C, "
          f"Avg: {stats['avg_C']}C, Delta: {stats['delta_C']}C")
    hotspot = grid.get_hotspot()
    print(f"  Hotspot: ({hotspot['x_mm']:.1f}, {hotspot['y_mm']:.1f})mm @ {hotspot['temp_C']}C")
    print()

    print("--- Thermal Map (50x50 -> ASCII) ---")
    print("  Legend: space(<30) .(+30) +(50) =(60) *(70) #(80) @(>90)")
    print(grid.thermal_map_str(25))
    print()

    # Compare with and without thermal vias
    print("--- With vs Without Thermal Vias ---")
    grid_no_via = ThermalGrid(50, 5.0)
    grid_no_via.add_source("MAC array", 1.0, 1.0, 3.0, 3.0, 2000)
    stats_nv = grid_no_via.step(1000)
    print(f"  Without vias: max={stats_nv['max_C']}C, avg={stats_nv['avg_C']}C")
    print(f"  With vias:    max={stats['max_C']}C, avg={stats['avg_C']}C")
    print(f"  Improvement: {stats_nv['max_C'] - stats['max_C']:.1f}C reduction")


if __name__ == "__main__":
    demo()
