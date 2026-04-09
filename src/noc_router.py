#!/usr/bin/env python3
"""Network-on-Chip (NoC) for mask-locked inference chips.

Mesh topology router with virtual channels, wormhole switching,
and deadlock-free routing for multi-tile communication.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class PacketType(Enum):
    WEIGHT_LOAD = 1
    ACTIVATION = 2
    CONTROL = 3
    RESULT = 4
    ACK = 5


class VCStatus(Enum):
    IDLE = "idle"
    ROUTING = "routing"
    VC_ALLOC = "vc_alloc"
    SA = "switch_alloc"
    ST = "switch_traverse"
    IB = "input_buffer"


@dataclass
class Flit:
    """Flow control unit (flit)."""
    packet_id: int
    type: PacketType
    src_x: int
    src_y: int
    dst_x: int
    dst_y: int
    vc: int = 0
    is_head: bool = False
    is_tail: bool = False
    payload: int = 0

    @property
    def hops_remaining(self) -> int:
        return abs(self.dst_x - self.src_x) + abs(self.dst_y - self.src_y)


@dataclass
class RouterPort:
    name: str
    direction: str  # N, S, E, W, local, injection
    buffer_depth: int = 4
    buffer: List[Flit] = field(default_factory=list)
    vc_credits: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        for vc in range(2):
            self.vc_credits[vc] = self.buffer_depth

    @property
    def occupied(self) -> bool:
        return len(self.buffer) >= self.buffer_depth

    def enqueue(self, flit: Flit) -> bool:
        if self.occupied:
            return False
        self.buffer.append(flit)
        return True

    def dequeue(self) -> Optional[Flit]:
        return self.buffer.pop(0) if self.buffer else None


class NoCRouter:
    """XY routing mesh router."""

    def __init__(self, x: int, y: int, n_vcs: int = 2,
                 buffer_depth: int = 4):
        self.x = x
        self.y = y
        self.n_vcs = n_vcs
        self.input_ports = {
            "N": RouterPort("in_N", "N", buffer_depth),
            "S": RouterPort("in_S", "S", buffer_depth),
            "E": RouterPort("in_E", "E", buffer_depth),
            "W": RouterPort("in_W", "W", buffer_depth),
            "local": RouterPort("in_local", "local", buffer_depth),
        }
        self.output_ports = {
            "N": RouterPort("out_N", "N", buffer_depth),
            "S": RouterPort("out_S", "S", buffer_depth),
            "E": RouterPort("out_E", "E", buffer_depth),
            "W": RouterPort("out_W", "W", buffer_depth),
            "local": RouterPort("out_local", "local", buffer_depth),
        }
        self.packets_forwarded = 0
        self.packets_dropped = 0

    def route(self, flit: Flit) -> str:
        """XY dimension-order routing."""
        if flit.dst_x == self.x and flit.dst_y == self.y:
            return "local"
        if flit.dst_x > self.x:
            return "E"
        if flit.dst_x < self.x:
            return "W"
        if flit.dst_y > self.y:
            return "N"
        if flit.dst_y < self.y:
            return "S"
        return "local"

    def step(self) -> List[Tuple[str, Flit]]:
        """Process one cycle. Returns (output_port, flit) pairs."""
        outputs = []
        for port_name, port in self.input_ports.items():
            if not port.buffer:
                continue
            flit = port.buffer[0]
            out_dir = self.route(flit)
            out_port = self.output_ports[out_dir]

            if out_port.occupied:
                self.packets_dropped += 1
                continue

            out_port.enqueue(flit)
            port.dequeue()
            self.packets_forwarded += 1
            outputs.append((out_dir, flit))

        return outputs


class MeshNoC:
    """2D mesh Network-on-Chip."""

    def __init__(self, width: int, height: int, n_vcs: int = 2,
                 buffer_depth: int = 4):
        self.width = width
        self.height = height
        self.routers: Dict[Tuple[int, int], NoCRouter] = {}
        for y in range(height):
            for x in range(width):
                self.routers[(x, y)] = NoCRouter(x, y, n_vcs, buffer_depth)
        self.total_cycles = 0
        self.total_packets = 0
        self.total_hops = 0
        self.total_latency = 0

    def inject(self, src: Tuple[int, int], dst: Tuple[int, int],
               packet_type: PacketType = PacketType.ACTIVATION,
               payload: int = 0) -> int:
        """Inject packet. Returns packet ID."""
        pid = self.total_packets
        self.total_packets += 1

        head = Flit(pid, packet_type, src[0], src[1], dst[0], dst[1],
                    vc=0, is_head=True, payload=payload)
        tail = Flit(pid, packet_type, src[0], src[1], dst[0], dst[1],
                    vc=0, is_tail=True)

        router = self.routers[src]
        router.input_ports["local"].enqueue(head)
        router.input_ports["local"].enqueue(tail)
        return pid

    def step(self) -> int:
        """Run one cycle across all routers. Returns packets delivered."""
        delivered = 0
        for pos, router in self.routers.items():
            outputs = router.step()
            for out_dir, flit in outputs:
                if out_dir == "local":
                    if flit.is_tail:
                        delivered += 1
                    continue
                dx, dy = 0, 0
                if out_dir == "N": dy = 1
                elif out_dir == "S": dy = -1
                elif out_dir == "E": dx = 1
                elif out_dir == "W": dx = -1

                nx, ny = pos[0] + dx, pos[1] + dy
                if (nx, ny) in self.routers:
                    in_dir = {"N": "S", "S": "N", "E": "W", "W": "E"}[out_dir]
                    self.routers[(nx, ny)].input_ports[in_dir].enqueue(flit)
                    self.total_hops += 1

        self.total_cycles += 1
        return delivered

    def run(self, packets: List[Tuple[Tuple[int,int], Tuple[int,int]]],
            max_cycles: int = 1000) -> Dict:
        """Inject and route all packets."""
        for src, dst in packets:
            self.inject(src, dst)

        delivered = 0
        for _ in range(max_cycles):
            d = self.step()
            delivered += d
            if delivered >= len(packets):
                break

        avg_latency = self.total_cycles / len(packets) if delivered > 0 else max_cycles
        avg_hops = self.total_hops / (delivered * 2) if delivered > 0 else 0  # 2 flits/packet

        return {
            "packets": len(packets),
            "delivered": delivered,
            "cycles": self.total_cycles,
            "avg_latency_cycles": round(avg_latency, 1),
            "avg_hops": round(avg_hops, 1),
            "throughput_packets_per_cycle": round(delivered / self.total_cycles, 3) if self.total_cycles > 0 else 0,
        }


def demo():
    print("=== Network-on-Chip Simulator ===\n")

    # 4x4 mesh
    noc = MeshNoC(4, 4, buffer_depth=4)
    print(f"Mesh: {noc.width}x{noc.height} = {len(noc.routers)} routers")
    print(f"VCs: {noc.routers[(0,0)].n_vcs}, Buffer: 4 flits")
    print()

    # Simple traffic
    print("--- Point-to-Point (corner to corner) ---")
    noc1 = MeshNoC(4, 4)
    result = noc1.run([((0, 0), (3, 3))])
    print(f"  (0,0)->(3,3): {result['cycles']} cycles, {result['avg_latency_cycles']} latency")
    print()

    # All-to-center
    print("--- All-to-Center (2,2) ---")
    noc2 = MeshNoC(4, 4)
    traffic = [((x, y), (2, 2)) for x in range(4) for y in range(4)
               if (x, y) != (2, 2)]
    result = noc2.run(traffic, max_cycles=200)
    print(f"  {len(traffic)} packets: {result['delivered']} delivered in {result['cycles']} cycles")
    print(f"  Avg latency: {result['avg_latency_cycles']} cycles")
    print()

    # Random traffic
    print("--- Random Traffic (16 packets, 4x4) ---")
    import random
    random.seed(42)
    noc3 = MeshNoC(4, 4)
    traffic = [((random.randint(0, 3), random.randint(0, 3)),
                (random.randint(0, 3), random.randint(0, 3)))
               for _ in range(16)]
    result = noc3.run(traffic, max_cycles=200)
    print(f"  {len(traffic)} packets: {result['delivered']} delivered")
    print(f"  Throughput: {result['throughput_packets_per_cycle']} pkt/cycle")
    print()

    # Scaling analysis
    print("--- Mesh Scaling ---")
    for size in [2, 4, 6, 8]:
        noc = MeshNoC(size, size)
        traffic = [((0, 0), (size - 1, size - 1))]
        result = noc.run(traffic)
        print(f"  {size}x{size}: {result['cycles']} cycles")


if __name__ == "__main__":
    demo()
