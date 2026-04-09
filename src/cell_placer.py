#!/usr/bin/env python3
"""Standard cell placer with latch-up awareness for mask-locked chips.

Row-based placement, tap insertion, guard ring generation,
and placement density optimization.
"""
import math, random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class PlacedCell:
    name: str
    lib_cell: str
    x: float  # um
    y: float  # um
    width: float
    height: float
    row: int
    col: int
    has_guard_ring: bool = False
    is_tap: bool = False


class CellPlacer:
    """Row-based standard cell placement."""

    def __init__(self, die_w: float = 5000, die_h: float = 5000,
                 row_height: float = 1.2, site_width: float = 0.2):
        self.die_w = die_w
        self.die_h = die_h
        self.row_h = row_height
        self.site_w = site_width
        self.margin = 200  # um from die edge

        # Available area
        self.avail_w = die_w - 2 * self.margin
        self.avail_h = die_h - 2 * self.margin
        self.n_cols = int(self.avail_w / site_width)
        self.n_rows = int(self.avail_h / row_height)

        self.placed: List[PlacedCell] = []
        self.row_occupancy: Dict[int, float] = {}  # row -> occupied width

    def place_instances(self, instances: Dict[str, Dict]) -> Dict:
        """Place cell instances in rows."""
        # instances: {name: {cell: "INV_X4", guard: bool, tap_every: int}}
        self.placed = []
        row = 0
        col = 0

        for inst_name, info in instances.items():
            cell_w = info.get("width", 1.0)
            cell_h = info.get("height", self.row_h)
            need_guard = info.get("guard", False)
            is_tap = info.get("is_tap", False)

            # Check if fits in current row
            avail = self.avail_w - self.row_occupancy.get(row, 0)
            if cell_w > avail:
                row += 1
                col = 0
                if row >= self.n_rows:
                    return {"error": f"Out of rows at {inst_name}"}

            x = self.margin + col * self.site_w
            y = self.margin + row * self.row_h

            self.placed.append(PlacedCell(
                inst_name, info.get("cell", "BUF_X1"),
                x, y, cell_w, cell_h, row, col,
                has_guard_ring=need_guard, is_tap=is_tap,
            ))
            self.row_occupancy[row] = self.row_occupancy.get(row, 0) + cell_w
            col += int(cell_w / self.site_w)

        return self._report()

    def _report(self) -> Dict:
        used_rows = set(p.row for p in self.placed)
        total_area = sum(p.width * p.height for p in self.placed)
        avail_area = self.avail_w * len(used_rows) * self.row_h if used_rows else 1
        utilization = total_area / avail_area * 100 if avail_area > 0 else 0

        taps = [p for p in self.placed if p.is_tap]
        guarded = [p for p in self.placed if p.has_guard_ring]

        return {
            "total_cells": len(self.placed),
            "used_rows": len(used_rows),
            "total_rows": self.n_rows,
            "utilization_pct": round(utilization, 1),
            "taps": len(taps),
            "guard_rings": len(guarded),
            "total_area_um2": round(total_area, 0),
            "bounding_box": self._bounding_box(),
        }

    def _bounding_box(self) -> Dict:
        if not self.placed:
            return {"w": 0, "h": 0}
        max_x = max(p.x + p.width for p in self.placed)
        max_y = max(p.y + p.height for p in self.placed)
        return {"w": round(max_x, 0), "h": round(max_y, 0)}

    def add_tap_cells(self, tap_interval_cols: int = 50) -> int:
        """Insert substrate tap cells at regular intervals."""
        taps_added = 0
        for row in range(self.n_rows):
            if row not in self.row_occupancy:
                continue
            last_tap_col = -tap_interval_cols
            for p in self.placed:
                if p.row == row and p.col - last_tap_col >= tap_interval_cols:
                    # Insert tap before this cell
                    tap = PlacedCell(
                        f"tap_r{row}_c{p.col}", "TAP_X1",
                        p.x - self.site_w, p.y,
                        self.site_w, self.row_h, row, p.col - 1,
                        is_tap=True,
                    )
                    self.placed.append(tap)
                    taps_added += 1
                    last_tap_col = p.col
        return taps_added

    def placement_map_str(self) -> str:
        """ASCII placement map."""
        grid_h = min(self.n_rows, 30)
        grid_w = min(self.n_cols, 60)
        grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]

        for p in self.placed:
            gy = int((p.y - self.margin) / self.row_h)
            gx_start = int((p.x - self.margin) / self.site_w)
            gx_end = min(int((p.x + p.width - self.margin) / self.site_w), grid_w)
            if 0 <= gy < grid_h:
                for gx in range(max(0, gx_start), gx_end):
                    if p.is_tap:
                        grid[gy][gx] = "T"
                    elif p.has_guard_ring:
                        grid[gy][gx] = "G"
                    elif "INV" in p.lib_cell:
                        grid[gy][gx] = "I"
                    elif "DFF" in p.lib_cell:
                        grid[gy][gx] = "D"
                    elif "BUF" in p.lib_cell:
                        grid[gy][gx] = "B"
                    else:
                        grid[gy][gx] = "X"

        return "\n".join("".join(row) for row in grid[:grid_h])


def demo():
    print("=== Cell Placer ===\n")

    placer = CellPlacer(5000, 5000, 1.2, 0.2)
    print(f"Die: {placer.die_w}x{placer.die_h}um, Rows: {placer.n_rows}, "
          f"Cols: {placer.n_cols}")
    print(f"Available: {placer.avail_w}x{placer.avail_h}um")
    print()

    # Generate instances (typical small design)
    instances = {}
    cell_types = [
        ("INV_X1", 0.4, False), ("INV_X2", 0.8, False),
        ("BUF_X2", 1.0, False), ("NAND2_X1", 0.8, False),
        ("DFF_X2", 2.0, False), ("FA_X1", 2.5, False),
        ("TAP_X1", 0.2, False),
    ]

    random.seed(42)
    for i in range(200):
        ct, w, guard = random.choice(cell_types)
        instances[f"u_{i}"] = {"cell": ct, "width": w,
                                "height": placer.row_h, "guard": guard}

    # Place
    result = placer.place_instances(instances)
    print("--- Placement Results ---")
    print(f"  Cells: {result['total_cells']}")
    print(f"  Rows used: {result['used_rows']}/{result['total_rows']}")
    print(f"  Utilization: {result['utilization_pct']}%")
    print(f"  Bounding box: {result['bounding_box']}")
    print()

    # Add taps
    taps = placer.add_tap_cells(30)
    print(f"  Taps inserted: {taps}")
    print()

    # Map
    print("--- Placement Map (I=INV, B=BUF, D=DFF, X=logic, T=tap, G=guard) ---")
    print(placer.placement_map_str()[:800])
    print()

    # Utilization targets
    print("--- Utilization vs Design Size ---")
    for n_cells in [50, 200, 500, 1000, 2000]:
        insts = {}
        for i in range(n_cells):
            ct, w, guard = random.choice(cell_types)
            insts[f"u_{i}"] = {"cell": ct, "width": w,
                                "height": placer.row_h, "guard": guard}
        p = CellPlacer(5000, 5000)
        r = p.place_instances(insts)
        print(f"  {n_cells:4d} cells: {r['used_rows']:2d} rows, "
              f"{r['utilization_pct']:5.1f}% util, "
              f"bbox={r['bounding_box']['w']}x{r['bounding_box']['h']}um")


if __name__ == "__main__":
    demo()
