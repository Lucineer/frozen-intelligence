#!/usr/bin/env python3
"""Clock tree synthesis for mask-locked chips.

Balanced H-tree clock distribution, buffer insertion,
skew analysis, and power estimation.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ClockNode:
    name: str
    x: float  # um
    y: float  # um
    level: int  # tree depth
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    is_leaf: bool = False
    buffer_type: str = "none"  # none, inv, buffer
    wire_length_um: float = 0
    arrival_time_ps: float = 0


@dataclass
class ClockTreeResult:
    frequency_mhz: float
    skew_ps: float
    insertion_delay_ps: float
    num_buffers: int
    total_wire_um: float
    clock_power_mw: float
    duty_cycle_pct: float


class ClockTreeSynthesizer:
    """H-tree clock tree synthesis and analysis."""

    def __init__(self, die_size_um: float = 5000, process_nm: int = 28):
        self.die = die_size_um
        self.nm = process_nm

        # Wire parameters
        self.wire_rc = {  # per um
            28: {"r": 0.5, "c": 0.2},    # ohm/um, fF/um
            40: {"r": 0.3, "c": 0.15},
            65: {"r": 0.2, "c": 0.12},
            130: {"r": 0.1, "c": 0.1},
        }
        self.rc = self.wire_rc.get(process_nm, self.wire_rc[28])

        # Buffer parameters
        self.buffer_delay_ps = 30  # inverter delay
        self.buffer_cap_ff = 5.0   # output capacitance fF
        self.buffer_power_nw = 0.5  # dynamic power per buffer (nW/MHz)
        self.max_fanout = 4

        self.nodes: Dict[str, ClockNode] = {}
        self.leaves: List[str] = []

    def build_htree(self, depth: int = 3) -> Dict[str, ClockNode]:
        """Build balanced H-tree clock distribution."""
        self.nodes = {}
        self.leaves = []

        cx, cy = self.die / 2, self.die / 2
        step = self.die / (2 ** (depth + 1))

        self._build_recursive("root", cx, cy, step, depth)
        return self.nodes

    def _build_recursive(self, name: str, x: float, y: float,
                          step: float, remaining: int, parent: str = None):
        self.nodes[name] = ClockNode(name, x, y, remaining, parent)

        if remaining == 0:
            self.nodes[name].is_leaf = True
            self.leaves.append(name)
            if parent:
                self.nodes[parent].children.append(name)
            return

        if parent:
            self.nodes[parent].children.append(name)

        # H-tree: branch in X then Y alternating by level
        if remaining % 2 == 1:
            offsets = [(-step, 0), (step, 0)]
        else:
            offsets = [(0, -step), (0, step)]

        for i, (dx, dy) in enumerate(offsets):
            child_name = f"{name}_{i}"
            child_step = step / 2
            self._build_recursive(child_name, x + dx, y + dy,
                                  child_step, remaining - 1, name)

        # Add buffer at each branch point
        if remaining > 1:
            self.nodes[name].buffer_type = "buffer"
        elif remaining == 1:
            self.nodes[name].buffer_type = "inv"

    def analyze(self, frequency_mhz: float = 500) -> ClockTreeResult:
        """Analyze clock tree timing and power."""
        # Wire lengths (parent to child)
        for name, node in self.nodes.items():
            if node.parent:
                parent = self.nodes[node.parent]
                dx = node.x - parent.x
                dy = node.y - parent.y
                node.wire_length_um = math.sqrt(dx*dx + dy*dy)

        # Recursive arrival time calculation
        self._compute_arrival("root", 0)

        # Skew = max arrival - min arrival at leaves
        leaf_arrivals = [self.nodes[l].arrival_time_ps for l in self.leaves]
        skew = max(leaf_arrivals) - min(leaf_arrivals) if leaf_arrivals else 0

        insertion_delay = self.nodes["root"].arrival_time_ps if self.leaves else 0

        # Power
        n_buf = sum(1 for n in self.nodes.values() if n.buffer_type != "none")
        total_wire = sum(n.wire_length_um for n in self.nodes.values())
        wire_power = total_wire * self.rc["c"] * 1e-15 * frequency_mhz * 1e6 * (0.9 ** 2)
        buf_power = n_buf * self.buffer_cap_ff * 1e-15 * frequency_mhz * 1e6 * (0.9 ** 2) * 0.5
        total_power = (wire_power + buf_power) * 1e6  # W -> uW

        # Duty cycle degradation through inverters
        n_inv = sum(1 for n in self.nodes.values() if n.buffer_type == "inv")
        duty = 50.0 - n_inv * 0.5  # slight degradation per inverter

        return ClockTreeResult(
            frequency_mhz=frequency_mhz,
            skew_ps=round(skew, 1),
            insertion_delay_ps=round(insertion_delay, 1),
            num_buffers=n_buf,
            total_wire_um=round(total_wire, 0),
            clock_power_mw=round(total_power * 1e-3, 3),  # uW -> mW
            duty_cycle_pct=round(duty, 1),
        )

    def _compute_arrival(self, name: str, arrival: float):
        node = self.nodes[name]
        node.arrival_time_ps = arrival

        if not node.children:
            return

        wire_delay = 0.5 * node.wire_length_um * self.rc["r"] * self.rc["c"] * 1e-3  # ps
        buf_delay = self.buffer_delay_ps if node.buffer_type != "none" else 0

        for child_name in node.children:
            child_arrival = arrival + wire_delay + buf_delay
            self._compute_arrival(child_name, child_arrival)

    def leaf_map_str(self) -> str:
        """ASCII map of leaf positions."""
        grid_size = 20
        grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

        for name in self.leaves:
            n = self.nodes[name]
            gx = min(int(n.x / self.die * grid_size), grid_size - 1)
            gy = min(int(n.y / self.die * grid_size), grid_size - 1)
            grid[gy][gx] = "X"

        # Mark root
        root = self.nodes["root"]
        rx = min(int(root.x / self.die * grid_size), grid_size - 1)
        ry = min(int(root.y / self.die * grid_size), grid_size - 1)
        grid[ry][rx] = "R"

        return "\n".join("".join(row) for row in grid)


def demo():
    print("=== Clock Tree Synthesizer ===\n")

    cts = ClockTreeSynthesizer(5000, 28)
    print(f"Die: {cts.die}um, Process: {cts.nm}nm")
    print(f"Wire RC: {cts.rc['r']} ohm/um, {cts.rc['c']} fF/um")
    print()

    # Build H-trees at different depths
    for depth in [1, 2, 3, 4]:
        cts.build_htree(depth)
        r = cts.analyze(500)
        print(f"--- Depth {depth}: {len(cts.leaves)} sinks, {r.num_buffers} buffers ---")
        print(f"  Skew: {r.skew_ps}ps, Insertion delay: {r.insertion_delay_ps}ps")
        print(f"  Wire: {r.total_wire_um:.0f}um, Power: {r.clock_power_mw:.3f}mW")
        print(f"  Duty cycle: {r.duty_cycle_pct}%")
        print()

    # Frequency sweep
    print("--- Frequency Sweep (depth=3) ---")
    cts.build_htree(3)
    for freq in [100, 250, 500, 1000, 2000]:
        r = cts.analyze(freq)
        print(f"  {freq:4d}MHz: skew={r.skew_ps:6.1f}ps, "
              f"power={r.clock_power_mw:.3f}mW, duty={r.duty_cycle_pct:.1f}%")

    # Leaf distribution
    print(f"\n--- Leaf Distribution (depth=3) ---")
    print(cts.leaf_map_str())

    # Process comparison
    print("\n--- Process Comparison (depth=3, 500MHz) ---")
    for nm in [28, 40, 65, 130]:
        c = ClockTreeSynthesizer(5000, nm)
        c.build_htree(3)
        r = c.analyze(500)
        print(f"  {nm}nm: skew={r.skew_ps:.1f}ps, wire={r.total_wire_um:.0f}um, "
              f"power={r.clock_power_mw:.3f}mW")


if __name__ == "__main__":
    demo()
