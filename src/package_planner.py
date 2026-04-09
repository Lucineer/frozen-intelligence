#!/usr/bin/env python3
"""Package substrate and wire bond planner for mask-locked chips.

Package selection, pad ring design, wire bond lengths,
and co-design constraints between die and package.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Pad:
    name: str
    x: float
    y: float
    pad_type: str    # signal, power, ground, analog
    side: str        # N, S, E, W
    bond_wire_um: float = 0  # computed wire length


@dataclass
class Package:
    name: str
    pitch_um: float     # pad pitch
    lead_count: int
    body_mm: float      # package body size
    die_max_mm: float   # max die size
    cavity_mm: float
    thermal_R_CW: float # thermal resistance junction to case


@dataclass
class PackageResult:
    package: Package
    pads_placed: int
    wire_lengths: Dict[str, float]
    max_wire_um: float
    avg_wire_um: float
    die_utilization: float
    pad_ring_pitch_um: float


PACKAGES = {
    "QFN32": Package("QFN32", 500, 32, 5.0, 3.0, 3.2, 30),
    "QFN48": Package("QFN48", 400, 48, 6.0, 4.0, 4.2, 25),
    "QFN64": Package("QFN64", 350, 64, 8.0, 5.0, 5.2, 20),
    "BGA64": Package("BGA64", 800, 64, 8.0, 5.0, 5.5, 15),
    "BGA128": Package("BGA128", 650, 128, 12.0, 8.0, 8.5, 10),
    "BGA256": Package("BGA256", 500, 256, 17.0, 12.0, 12.5, 8),
    "BGA384": Package("BGA384", 450, 384, 21.0, 15.0, 15.5, 6),
    "WLCSP36": Package("WLCSP36", 400, 36, 3.5, 2.5, 2.7, 35),
}


class PackagePlanner:
    """Die/package co-design planner."""

    def __init__(self, die_size_um: float = 5000):
        self.die = die_size_um
        self.pad_width = 80   # um
        self.pad_height = 80  # um
        self.pad_gap = 40     # um minimum gap
        self.ring_margin = 150  # um from die edge to pad ring

    def max_pads_per_side(self) -> int:
        usable = self.die - 2 * self.ring_margin
        pitch = self.pad_width + self.pad_gap
        return int(usable / pitch)

    def total_pad_capacity(self) -> int:
        per_side = self.max_pads_per_side()
        return per_side * 4  # 4 sides

    def design_pad_ring(self, n_signal: int, n_power: int = 4,
                        n_ground: int = 8) -> Dict:
        """Design pad ring layout."""
        per_side = self.max_pads_per_side()
        total_pads = n_signal + n_power + n_ground
        pads = []

        # Distribute power/ground evenly on all sides
        pg_per_side = (n_power + n_ground) // 4
        sig_per_side = (per_side - pg_per_side)

        for side in ["N", "S", "E", "W"]:
            offset = self.ring_margin + self.pad_width / 2

            for i in range(per_side):
                pos = offset + i * (self.pad_width + self.pad_gap)
                if side == "N":
                    x, y = pos, self.ring_margin
                elif side == "S":
                    x, y = pos, self.die - self.ring_margin
                elif side == "E":
                    x, y = self.die - self.ring_margin, pos
                else:  # W
                    x, y = self.ring_margin, pos

                pad_type = "signal"
                if i < pg_per_side // 2:
                    pad_type = "ground"
                elif i == per_side - 1:
                    pad_type = "power"

                pad = Pad(f"{side}_{i}", x, y, pad_type, side)
                pads.append(pad)

        # Compute wire bond lengths (to nearest package lead)
        pkg_pitch = 500
        for pad in pads:
            if pad.side == "N":
                bond = self.die - pad.y
            elif pad.side == "S":
                bond = pad.y
            elif pad.side == "E":
                bond = self.die - pad.x
            else:
                bond = pad.x
            pad.bond_wire_um = bond * 1.5  # wire goes out and down

        lengths = [p.bond_wire_um for p in pads]
        return {
            "pads": pads,
            "total": len(pads),
            "signal": sum(1 for p in pads if p.pad_type == "signal"),
            "power": sum(1 for p in pads if p.pad_type == "power"),
            "ground": sum(1 for p in pads if p.pad_type == "ground"),
            "per_side": per_side,
            "pitch_um": self.pad_width + self.pad_gap,
            "max_wire_um": max(lengths),
            "avg_wire_um": sum(lengths) / len(lengths) if lengths else 0,
        }

    def select_package(self, n_pads: int, thermal_budget: float = 25) -> Dict:
        """Select optimal package for pad count and thermal."""
        candidates = []
        for name, pkg in PACKAGES.items():
            if pkg.lead_count >= n_pads and pkg.die_max_mm * 1000 >= self.die:
                candidates.append((name, pkg))

        results = []
        for name, pkg in candidates:
            utilization = n_pads / pkg.lead_count * 100
            die_util = (self.die / 1000 / pkg.die_max_mm) * 100
            results.append({
                "package": name,
                "lead_count": pkg.lead_count,
                "utilization_pct": round(utilization, 1),
                "die_util_pct": round(die_util, 1),
                "thermal_R": pkg.thermal_R_CW,
                "body_mm": pkg.body_mm,
                "max_die_mm": pkg.die_max_mm,
                "thermal_ok": pkg.thermal_R_CW <= thermal_budget,
                "score": round(utilization * 0.5 + (100 - die_util) * 0.3 +
                             (25 if pkg.thermal_R_CW <= thermal_budget else 0), 1),
            })

        results.sort(key=lambda x: -x["score"])
        return results


def demo():
    print("=== Package Planner ===\n")

    planner = PackagePlanner(5000)
    print(f"Die: {planner.die}um ({planner.die/1000}mm)")
    print(f"Max pads/side: {planner.max_pads_per_side()}, Total: {planner.total_pad_capacity()}")
    print()

    # Pad ring
    print("--- Pad Ring Design ---")
    ring = planner.design_pad_ring(40, 4, 8)
    print(f"  Total: {ring['total']}, Signal: {ring['signal']}, "
          f"Power: {ring['power']}, Ground: {ring['ground']}")
    print(f"  Pitch: {ring['pitch_um']}um, Per side: {ring['per_side']}")
    print(f"  Wire: max={ring['max_wire_um']:.0f}um, avg={ring['avg_wire_um']:.0f}um")
    print()

    # Package selection
    print("--- Package Selection (40 pads, 25 C/W thermal budget) ---")
    for r in planner.select_package(40, 25):
        therm = "OK" if r["thermal_ok"] else "FAIL"
        print(f"  {r['package']:8s}: {r['lead_count']:3d} leads, "
              f"util={r['utilization_pct']:4.0f}%, die={r['die_util_pct']:3.0f}%, "
              f"R={r['thermal_R']}C/W [{therm}] score={r['score']}")

    # Die size sweep
    print("\n--- Pad Capacity vs Die Size ---")
    for die_um in [2000, 3000, 4000, 5000, 6000, 8000]:
        p = PackagePlanner(die_um)
        print(f"  {die_um/1000:.1f}mm: {p.max_pads_per_side():3d}/side = "
              f"{p.total_pad_capacity()} total pads")

    # Available packages
    print("\n--- All Packages ---")
    for name, pkg in PACKAGES.items():
        print(f"  {name:8s}: {pkg.lead_count:3d} leads, {pkg.pitch_um}um pitch, "
              f"body={pkg.body_mm}mm, max_die={pkg.die_max_mm}mm, "
              f"R={pkg.thermal_R_CW}C/W")


if __name__ == "__main__":
    demo()
