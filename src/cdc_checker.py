#!/usr/bin/env python3
"""Clock domain crossing (CDC) checker for mask-locked chips.

Detects unsafe crossings, validates synchronizer chains,
and generates async FIFO Verilog.
"""
import re, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set


@dataclass
class Signal:
    name: str
    width: int = 1
    domain: str = "clk"
    direction: str = "output"
    is_clock: bool = False


@dataclass
class CDCViolation:
    kind: str        # unsync_cross, async_reset, glitch, metastable
    signal: str
    src_domain: str
    dst_domain: str
    severity: str    # ERROR, WARNING, INFO
    description: str


@dataclass
class SyncChain:
    signal: str
    src_domain: str
    dst_domain: str
    stages: int
    chain_type: str  # ff_sync, async_fifo, gray_counter, handshake
    is_safe: bool


class CDCChecker:
    """Detect and validate clock domain crossings."""

    def __init__(self):
        self.signals: Dict[str, Signal] = {}
        self.violations: List[CDCViolation] = []
        self.sync_chains: List[SyncChain] = []
        self.domains: Set[str] = set()

    def add_signal(self, name: str, domain: str = "clk",
                   width: int = 1, direction: str = "output",
                   is_clock: bool = False):
        self.signals[name] = Signal(name, width, domain, direction, is_clock)
        if is_clock:
            self.domains.add(name)

    def register_sync_chain(self, signal: str, src: str, dst: str,
                            stages: int = 2, chain_type: str = "ff_sync"):
        chain = SyncChain(signal, src, dst, stages, chain_type,
                         stages >= 2)
        self.sync_chains.append(chain)
        return chain

    def parse_verilog_cdc(self, lines: List[str]) -> List[CDCViolation]:
        """Parse Verilog for CDC patterns."""
        violations = []
        synced_signals = set()

        # Detect sync chains: always @(posedge clk_b) ... <= sig_a;
        sync_pattern = re.compile(
            r"always\s*@\(posedge\s+(\w+)\).*?"
            r"(\w+)\s*<=\s*(\w+)", re.DOTALL)

        # Detect async FIFO patterns
        fifo_pattern = re.compile(r"(gray|async_fifo|cdc_fifo|sync_fifo)", re.I)

        # Detect raw crossings without sync
        for line in lines:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Check for assign crossing domains
            assign_m = re.match(r"assign\s+(\w+)\s*=\s*(\w+)", line)
            if assign_m:
                dst_sig = assign_m.group(1)
                src_sig = assign_m.group(2)
                if dst_sig in self.signals and src_sig in self.signals:
                    dst_d = self.signals[dst_sig].domain
                    src_d = self.signals[src_sig].domain
                    if dst_d != src_d:
                        # Check if synced
                        is_synced = any(
                            c.signal == src_sig and c.dst_domain == dst_d
                            for c in self.sync_chains
                        )
                        if not is_synced:
                            violations.append(CDCViolation(
                                "unsync_cross", src_sig, src_d, dst_d,
                                "ERROR",
                                f"assign {dst_sig} = {src_sig} crosses "
                                f"{src_d} -> {dst_d} without synchronizer"
                            ))

        self.violations.extend(violations)
        return violations

    def check(self) -> Dict:
        """Run full CDC analysis."""
        violations = []

        # Check all sync chains
        for chain in self.sync_chains:
            if chain.stages < 2 and chain.chain_type == "ff_sync":
                violations.append(CDCViolation(
                    "metastable", chain.signal, chain.src_domain,
                    chain.dst_domain, "ERROR",
                    f"{chain.signal}: only {chain.stages}-stage sync "
                    f"(minimum 2 for reliable metastability resolution)"
                ))

            # Multi-bit buses need Gray coding or async FIFO
            sig = self.signals.get(chain.signal)
            if sig and sig.width > 1 and chain.chain_type == "ff_sync":
                violations.append(CDCViolation(
                    "glitch", chain.signal, chain.src_domain,
                    chain.dst_domain, "WARNING",
                    f"{chain.signal}: {sig.width}-bit bus uses simple "
                    f"FF sync (risk of bit skew, consider Gray/FIFO)"
                ))

        # Check for clock signals used as data
        for name, sig in self.signals.items():
            if sig.is_clock and sig.direction == "output":
                for chain in self.sync_chains:
                    if chain.signal == name:
                        violations.append(CDCViolation(
                            "unsync_cross", name, sig.domain,
                            chain.dst_domain, "INFO",
                            f"Clock {name} crossing to {chain.dst_domain} "
                            f"(ensure clock divider or toggle sync)"
                        ))

        self.violations = violations
        return {
            "total_violations": len(violations),
            "errors": len([v for v in violations if v.severity == "ERROR"]),
            "warnings": len([v for v in violations if v.severity == "WARNING"]),
            "sync_chains": len(self.sync_chains),
            "domains": list(self.domains),
            "violations": violations,
        }

    def generate_async_fifo(self, depth: int = 16, data_width: int = 8,
                            wr_clk: str = "wr_clk", rd_clk: str = "rd_clk",
                            name: str = "async_fifo") -> str:
        """Generate async FIFO Verilog with Gray pointer sync."""
        W = data_width
        addr_bits = int(math.log2(depth))
        ptr_bits = addr_bits + 1  # extra bit for full/empty
        lines = []

        lines.append(f"module {name} (")
        lines.append(f"    input {wr_clk}, input wr_rst_n,")
        lines.append(f"    input wr_en, input [{W-1}:0] wr_data,")
        lines.append(f"    output wr_full,")
        lines.append(f"    input {rd_clk}, input rd_rst_n,")
        lines.append(f"    input rd_en,")
        lines.append(f"    output [{W-1}:0] rd_data,")
        lines.append(f"    output rd_empty")
        lines.append(f");")

        # Memory
        lines.append(f"    reg [{W-1}:0] mem [0:{depth-1}];")

        # Pointers
        lines.append(f"    reg [{ptr_bits-1}:0] wr_ptr, rd_ptr;")
        lines.append(f"    reg [{ptr_bits-1}:0] wr_ptr_gray, rd_ptr_gray;")
        lines.append(f"    reg [{ptr_bits-1}:0] wr_ptr_gray_sync1, wr_ptr_gray_sync2;")
        lines.append(f"    reg [{ptr_bits-1}:0] rd_ptr_gray_sync1, rd_ptr_gray_sync2;")

        # Binary to Gray
        lines.append(f"    function [{ptr_bits-1}:0] bin2gray;")
        lines.append(f"        input [{ptr_bits-1}:0] bin;")
        lines.append(f"        integer i;")
        lines.append(f"        begin")
        lines.append(f"            bin2gray = bin ^ (bin >> 1);")
        lines.append(f"        end")
        lines.append(f"    endfunction")

        # Write side
        lines.append(f"    always @(posedge {wr_clk} or negedge wr_rst_n) begin")
        lines.append(f"        if (!wr_rst_n) begin")
        lines.append(f"            wr_ptr <= 0; wr_ptr_gray <= 0;")
        lines.append(f"        end else if (wr_en && !wr_full) begin")
        lines.append(f"            mem[wr_ptr[addr_bits-1:0]] <= wr_data;")
        lines.append(f"            wr_ptr <= wr_ptr + 1;")
        lines.append(f"            wr_ptr_gray <= bin2gray(wr_ptr + 1);")
        lines.append(f"        end")
        lines.append(f"    end")

        # Read side
        lines.append(f"    always @(posedge {rd_clk} or negedge rd_rst_n) begin")
        lines.append(f"        if (!rd_rst_n) begin")
        lines.append(f"            rd_ptr <= 0; rd_ptr_gray <= 0;")
        lines.append(f"        end else if (rd_en && !rd_empty) begin")
        lines.append(f"            rd_ptr <= rd_ptr + 1;")
        lines.append(f"            rd_ptr_gray <= bin2gray(rd_ptr + 1);")
        lines.append(f"        end")
        lines.append(f"    end")

        # Sync pointers (2-stage FF sync)
        lines.append(f"    always @(posedge {wr_clk}) begin")
        lines.append(f"        rd_ptr_gray_sync1 <= rd_ptr_gray;")
        lines.append(f"        rd_ptr_gray_sync2 <= rd_ptr_gray_sync1;")
        lines.append(f"    end")
        lines.append(f"    always @(posedge {rd_clk}) begin")
        lines.append(f"        wr_ptr_gray_sync1 <= wr_ptr_gray;")
        lines.append(f"        wr_ptr_gray_sync2 <= wr_ptr_gray_sync1;")
        lines.append(f"    end")

        # Full/empty flags
        ones = "1'b1, " + "'d0, " * (ptr_bits - 2) + "'d0" if ptr_bits > 2 else "1'b1"
        full_check = "(rd_ptr_gray_sync2 ^ {" + ones + "})"
        lines.append("    assign wr_full = (wr_ptr_gray == " + full_check + ");")
        lines.append(f"    assign rd_empty = (rd_ptr_gray == wr_ptr_gray_sync2);")
        lines.append(f"    assign rd_data = mem[rd_ptr[addr_bits-1:0]];")

        lines.append(f"endmodule")
        return "\n".join(lines)


def demo():
    print("=== CDC Checker ===\n")

    cdc = CDCChecker()

    # Define signals in two clock domains
    for i in range(8):
        cdc.add_signal(f"data_{i}", "sys_clk", 1, "output")
    cdc.add_signal("data_bus", "sys_clk", 8, "output")
    cdc.add_signal("status", "sys_clk", 1, "output")
    cdc.add_signal("config", "pll_clk", 1, "output")
    cdc.add_signal("pll_clk", "pll_clk", 1, "clock", is_clock=True)

    # Register sync chains
    cdc.register_sync_chain("status", "sys_clk", "audio_clk", 2, "ff_sync")
    cdc.register_sync_chain("data_0", "sys_clk", "audio_clk", 1, "ff_sync")  # unsafe!
    cdc.register_sync_chain("data_bus", "sys_clk", "audio_clk", 2, "ff_sync")  # multi-bit warning

    # Check
    result = cdc.check()
    print(f"--- CDC Analysis ---")
    print(f"  Sync chains: {result['sync_chains']}")
    print(f"  Violations: {result['total_violations']} "
          f"({result['errors']} errors, {result['warnings']} warnings)")
    print()
    for v in result["violations"]:
        print(f"  [{v.severity}] {v.kind}: {v.signal}")
        print(f"    {v.description}")
        print()

    # Generate async FIFO
    print("--- Async FIFO (16x8) ---")
    fifo = cdc.generate_async_fifo(16, 8)
    print(fifo)
    print(f"\n  Generated: {len(fifo)} chars")


if __name__ == "__main__":
    demo()
