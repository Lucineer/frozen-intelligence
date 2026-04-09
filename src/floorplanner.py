#!/usr/bin/env python3
"""Chip floorplanner for mask-locked inference chips.

Generates die floorplans: weight bank placement, I/O pad ring,
power grid, clock tree, and thermal hot spot analysis.
"""
import math, json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Rectangle:
    x: float  # um
    y: float  # um
    w: float  # um
    h: float  # um
    name: str = ""

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    def overlaps(self, other: 'Rectangle') -> bool:
        return not (self.right <= other.x or other.right <= self.x or
                    self.top <= other.y or other.top <= self.y)

    def distance_to(self, other: 'Rectangle') -> float:
        dx = max(other.x - self.right, self.x - other.right, 0)
        dy = max(other.y - self.top, self.y - other.top, 0)
        return math.sqrt(dx * dx + dy * dy)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.top


class PowerGrid:
    """Power delivery network."""

    def __init__(self, die: Rectangle, vdd: float = 0.9, rings: int = 2,
                 stripe_pitch_um: float = 50.0):
        self.die = die
        self.vdd = vdd
        self.rings = rings
        self.stripe_pitch = stripe_pitch_um
        self.ir_drop_target = vdd * 0.05  # 5% IR drop budget

    def generate(self) -> Dict:
        """Generate power grid layout."""
        rings = []
        for i in range(self.rings):
            margin = 10 + i * 5
            rings.append(Rectangle(
                self.die.x + margin, self.die.y + margin,
                self.die.w - 2 * margin, self.die.h - 2 * margin,
                f"power_ring_{i}"))

        vstripes = []
        hstripes = []
        x = self.die.x + 20
        while x < self.die.right - 20:
            vstripes.append({"x": x, "y1": self.die.y + 15, "y2": self.die.top - 15,
                             "name": f"vstripe_{x:.0f}"})
            x += self.stripe_pitch

        y = self.die.y + 20
        while y < self.die.top - 20:
            hstripes.append({"y": y, "x1": self.die.x + 15, "x2": self.die.right - 15,
                             "name": f"hstripe_{y:.0f}"})
            y += self.stripe_pitch

        return {"vdd": self.vdd, "rings": len(rings),
                "v_stripes": len(vstripes), "h_stripes": len(hstripes),
                "ir_drop_budget_mv": round(self.ir_drop_target * 1000, 1)}


class ClockTree:
    """Clock distribution network."""

    def __init__(self, die: Rectangle, target_skew_ps: float = 50.0):
        self.die = die
        self.target_skew = target_skew_ps

    def generate(self, sinks: List[Tuple[float, float]]) -> Dict:
        """Generate balanced clock tree."""
        if not sinks:
            return {"stages": 0, "buffers": 0, "skew_ps": 0, "wire_len_um": 0}

        cx = sum(s[0] for s in sinks) / len(sinks)
        cy = sum(s[1] for s in sinks) / len(sinks)
        root = (cx, cy)

        max_dist = max(math.sqrt((s[0] - cx) ** 2 + (s[1] - cy) ** 2) for s in sinks)
        stages = max(1, int(math.log2(max(max_dist / 100, 1)))) + 1
        buffers = len(sinks) * stages
        wire_len = max_dist * 2 * stages

        return {"root": (round(cx, 1), round(cy, 1)),
                "stages": stages, "buffers": buffers,
                "skew_ps": round(self.target_skew, 1),
                "wire_len_um": round(wire_len, 1),
                "max_fanout": max(4, len(sinks) // stages)}


class ThermalAnalyzer:
    """Thermal hot spot analysis."""

    def __init__(self, die: Rectangle, max_temp_c: float = 100.0):
        self.die = die
        self.max_temp = max_temp_c
        self.ambient = 45.0

    def analyze(self, blocks: List[Rectangle]) -> Dict:
        """Analyze thermal distribution from block power density."""
        total_power = sum(b.area * 0.001 for b in blocks)  # 1mW/um2
        die_area = self.die.area
        avg_density = total_power / die_area * 1e6 if die_area > 0 else 0

        hotspots = []
        for b in blocks:
            density = b.area * 0.001 / b.area * 1e6  # W/m2
            temp_rise = density * 0.01  # simplified thermal resistance
            temp = self.ambient + temp_rise
            if temp > self.max_temp * 0.8:
                hotspots.append({"name": b.name, "temp_c": round(temp, 1),
                                 "x": b.cx, "y": b.cy, "severity": "warning"})

        return {"total_power_mw": round(total_power * 1000, 1),
                "avg_power_density": round(avg_density, 2),
                "hotspots": hotspots,
                "thermal_margin_c": round(self.max_temp - self.ambient, 1)}


class Floorplanner:
    """Chip floorplan generator."""

    def __init__(self, die_size_um: float = 2000.0, process_nm: int = 28):
        self.die = Rectangle(0, 0, die_size_um, die_size_um, "die")
        self.process_nm = process_nm
        self.pad_width = 80.0
        self.pad_pitch = 100.0
        self.blocks: List[Rectangle] = []

    def place_weight_banks(self, num_layers: int, tile_size_um: float = 100.0) -> List[Rectangle]:
        """Place weight bank tiles in core area."""
        margin = self.pad_width + 20
        core = Rectangle(margin, margin,
                         self.die.w - 2 * margin, self.die.h - 2 * margin, "core")

        tiles = []
        placed = 0
        y = core.y
        while y + tile_size_um <= core.top and placed < num_layers:
            x = core.x
            while x + tile_size_um <= core.right and placed < num_layers:
                tile = Rectangle(x, y, tile_size_um, tile_size_um,
                                 f"wb_{placed}")
                tiles.append(tile)
                self.blocks.append(tile)
                placed += 1
                x += tile_size_um + 5  # 5um gap
            y += tile_size_um + 5

        return tiles

    def place_io_pads(self, pad_count: int) -> List[Dict]:
        """Place I/O pads around die perimeter."""
        pads = []
        perimeter = 2 * (self.die.w + self.die.h)
        spacing = perimeter / pad_count

        pos = 0.0
        for i in range(pad_count):
            t = pos / perimeter
            # Walk around perimeter
            perim = self.die.w + self.die.h * 2 + self.die.w
            p = t * perim

            if p < self.die.w:
                px, py = p, 0
                side = "bottom"
            elif p < self.die.w + self.die.h:
                px, py = self.die.w, p - self.die.w
                side = "right"
            elif p < 2 * self.die.w + self.die.h:
                px, py = self.die.w - (p - self.die.w - self.die.h), self.die.h
                side = "top"
            else:
                px, py = 0, self.die.h - (p - 2 * self.die.w - self.die.h)
                side = "left"

            pads.append({"id": i, "x": round(px, 1), "y": round(py, 1),
                         "side": side, "type": "io"})
            pos += spacing

        return pads

    def place_mac_array(self, rows: int, cols: int,
                        tile_um: float = 200.0) -> Rectangle:
        """Place systolic MAC array in center."""
        total_w = cols * tile_um
        total_h = rows * tile_um
        x = (self.die.w - total_w) / 2
        y = (self.die.h - total_h) / 2
        mac = Rectangle(x, y, total_w, total_h, "mac_array")
        self.blocks.append(mac)
        return mac

    def generate_floorplan(self, n_layers: int = 24, n_heads: int = 8,
                           mac_rows: int = 4, mac_cols: int = 4,
                           pad_count: int = 40) -> Dict:
        """Generate complete floorplan."""
        # Place components
        weight_tiles = self.place_weight_banks(n_layers)
        mac = self.place_mac_array(mac_rows, mac_cols)
        pads = self.place_io_pads(pad_count)

        # Place control logic
        ctrl = Rectangle(10, self.die.h - 90, 120, 80, "control")
        self.blocks.append(ctrl)

        # Place clock
        clk = Rectangle(self.die.w / 2 - 30, self.die.h - 60, 60, 50, "pll")
        self.blocks.append(clk)

        # Power grid
        pg = PowerGrid(self.die, vdd=0.9 if self.process_nm <= 28 else 1.2)
        power = pg.generate()

        # Clock tree
        sinks = [(b.cx, b.cy) for b in self.blocks]
        ct = ClockTree(self.die)
        clock = ct.generate(sinks)

        # Thermal analysis
        ta = ThermalAnalyzer(self.die)
        thermal = ta.analyze(self.blocks)

        # Area utilization
        block_area = sum(b.area for b in self.blocks)
        utilization = block_area / self.die.area * 100

        return {
            "die_size_um": self.die.w,
            "process_nm": self.process_nm,
            "weight_tiles": len(weight_tiles),
            "mac_array": {"x": round(mac.x, 1), "y": round(mac.y, 1),
                          "w": round(mac.w, 1), "h": round(mac.h, 1)},
            "pads": len(pads),
            "blocks": len(self.blocks),
            "utilization_pct": round(utilization, 1),
            "power": power,
            "clock": clock,
            "thermal": thermal,
        }


def demo():
    print("=== Chip Floorplanner ===\n")

    fp = Floorplanner(die_size_um=2000.0, process_nm=28)
    plan = fp.generate_floorplan(n_layers=24, n_heads=8, pad_count=40)

    print(f"Die: {plan['die_size_um']}um, Process: {plan['process_nm']}nm")
    print(f"Weight tiles: {plan['weight_tiles']}")
    print(f"MAC array: {plan['mac_array']['w']}x{plan['mac_array']['h']}um "
          f"at ({plan['mac_array']['x']}, {plan['mac_array']['y']})")
    print(f"I/O pads: {plan['pads']}")
    print(f"Blocks: {plan['blocks']}")
    print(f"Utilization: {plan['utilization_pct']}%")
    print()

    print("--- Power Grid ---")
    p = plan["power"]
    print(f"  VDD: {p['vdd']}V, Rings: {p['rings']}")
    print(f"  Stripes: {p['v_stripes']}V + {p['h_stripes']}H")
    print(f"  IR drop budget: {p['ir_drop_budget_mv']}mV")
    print()

    print("--- Clock Tree ---")
    c = plan["clock"]
    print(f"  Root: {c['root']}")
    print(f"  Stages: {c['stages']}, Buffers: {c['buffers']}")
    print(f"  Skew: {c['skew_ps']}ps, Wire: {c['wire_len_um']}um")
    print()

    print("--- Thermal ---")
    t = plan["thermal"]
    print(f"  Power: {t['total_power_mw']}mW")
    print(f"  Hotspots: {len(t['hotspots'])}")
    print()

    # Compare processes
    print("--- Process Comparison ---")
    for nm in [28, 40, 65, 130]:
        f = Floorplanner(2000, nm)
        p = f.generate_floorplan(24, 8)
        print(f"  {nm:3d}nm: {p['utilization_pct']}% utilization")


if __name__ == "__main__":
    demo()
