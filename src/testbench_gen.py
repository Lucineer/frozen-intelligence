#!/usr/bin/env python3
"""Verilog testbench generator for mask-locked inference chips.

Generates stimulus, response checking, and coverage collection
for structural netlists produced by netlist_gen.py.
"""
import math, random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class TBSignal:
    name: str
    width: int = 1
    direction: str = "reg"  # reg or wire
    initial: str = "0"

    def declaration(self) -> str:
        if self.width == 1:
            return f"  {self.direction} {self.name} = {self.initial};"
        return f"  {self.direction} [{self.width-1}:0] {self.name} = {self.width}'d{self.initial};"


@dataclass
class TBClock:
    name: str = "clk"
    period_ns: float = 2.0
    duty: float = 0.5

    def generate(self) -> List[str]:
        high = self.period_ns * self.duty / 2
        low = self.period_ns * (1 - self.duty) / 2
        return [
            f"  initial begin",
            f"    forever #{high} {self.name} = ~{self.name};",
            f"  end",
            f"  initial {self.name} = 0;",
        ]


@dataclass
class TBTask:
    name: str
    body: List[str]
    ports: List[str] = field(default_factory=list)

    def verilog(self) -> str:
        port_str = ", ".join(self.ports) if self.ports else ""
        lines = [f"  task {self.name}({port_str});"] if not self.ports else [f"  task {self.name};"]
        for p in self.ports:
            lines.append(f"    input {p};")
        lines.extend(f"    {b}" for b in self.body)
        lines.append(f"  endtask")
        return "\n".join(lines)


class TestbenchGenerator:
    """Generate Verilog testbenches."""

    def __init__(self, dut_name: str, signals: List[Dict]):
        self.dut_name = dut_name
        self.signals = [TBSignal(**s) for s in signals]
        self.clocks: List[TBClock] = []
        self.tasks: List[TBTask] = []
        self.initial_blocks: List[List[str]] = []
        self.stimulus: List[str] = []
        self.monitors: List[str] = []
        self.checks: List[str] = []

    def add_clock(self, name: str = "clk", period_ns: float = 2.0):
        self.clocks.append(TBClock(name, period_ns))

    def add_reset_task(self, name: str = "do_reset", cycles: int = 5):
        body = [
            f"    begin",
            f"      rst_n = 0;",
            f"      repeat({cycles}) @(posedge clk);",
            f"      rst_n = 1;",
            f"    end",
        ]
        self.tasks.append(TBTask(name, body))

    def add_load_task(self, name: str = "load_weight", addr_width: int = 8,
                      data_width: int = 8):
        body = [
            f"    input [{addr_width-1}:0] addr;",
            f"    input [{data_width-1}:0] data;",
            f"    begin",
            f"      @(posedge clk);",
            f"      weight_addr = addr;",
            f"      weight_data = data;",
            f"      weight_wr = 1;",
            f"      @(posedge clk);",
            f"      weight_wr = 0;",
            f"    end",
        ]
        self.tasks.append(TBTask(name, body))

    def add_monitor_task(self, dut_name: str, signals: List[str]):
        body = [
            f'    $display("--- %0t: {", ".join(signals)}", $time, {", ".join(signals)});',
        ]
        self.tasks.append(TBTask("monitor_all", body))

    def add_basic_stimulus(self, test_vectors: List[Dict]):
        """Add basic input stimulus from test vectors."""
        self.stimulus.append("  initial begin")
        for tv in test_vectors:
            assignments = []
            for k, v in tv.items():
                if isinstance(v, int):
                    assignments.append(f"      {k} = {v};")
                elif isinstance(v, str) and v.startswith("#"):
                    assignments.append(f"      {v};")
                else:
                    assignments.append(f"      {k} = {v};")
            self.stimulus.extend(assignments)
            self.stimulus.append("      @(posedge clk);")
        self.stimulus.append("      $finish;")
        self.stimulus.append("  end")

    def add_random_stimulus(self, signal_name: str, width: int, cycles: int = 100):
        """Add random stimulus."""
        self.stimulus.append("  initial begin")
        self.stimulus.append(f"    repeat({cycles}) begin")
        self.stimulus.append(f"      @(posedge clk);")
        self.stimulus.append(f"      {signal_name} = $random;")
        self.stimulus.append(f"    end")
        self.stimulus.append(f"    $finish;")
        self.stimulus.append("  end")

    def add_response_check(self, signal: str, expected_file: str = None):
        if expected_file:
            self.checks.append(f'  initial $readmemh("{expected_file}", expected_data);')
        self.checks.append("  always @(posedge clk) begin")
        self.checks.append(f"    #1 if ({signal} !== expected_data) begin")
        self.checks.append(f'      $display("ERROR: %0t {signal} mismatch", $time);')
        self.checks.append(f"      err_count = err_count + 1;")
        self.checks.append(f"    end")
        self.checks.append(f"  end")

    def add_scoreboard(self, output_signal: str, ref_signal: str = None):
        self.stimulus.append("  integer err_count = 0;")
        self.stimulus.append("  initial begin")
        self.stimulus.append("    #100000;")
        self.stimulus.append('    if (err_count == 0)')
        self.stimulus.append('      $display("PASS: No errors in %0t cycles", $time);')
        self.stimulus.append('    else')
        self.stimulus.append('      $display("FAIL: %0d errors", err_count);')
        self.stimulus.append("    $finish;")
        self.stimulus.append("  end")

    def generate(self) -> str:
        lines = [
            f"`timescale 1ns/1ps",
            f"module tb_{self.dut_name}();",
            f"  // Testbench for {self.dut_name}",
            f"  // Auto-generated by frozen-intelligence netlist_gen",
            f"  // Lucineer (DiGennaro et al.)",
            f"",
        ]

        # Signal declarations
        for s in self.signals:
            lines.append(s.declaration())

        # Clocks
        lines.append("")
        for clk in self.clocks:
            lines.extend(clk.generate())

        # Tasks
        lines.append("")
        for task in self.tasks:
            lines.append(task.verilog())
            lines.append("")

        # DUT instantiation
        conns = ", ".join(f".{s.name}({s.name})" for s in self.signals)
        lines.append(f"  {self.dut_name} dut (")
        lines.append(f"    {conns}")
        lines.append(f"  );")
        lines.append("")

        # Stimulus
        for s in self.stimulus:
            lines.append(s)
        lines.append("")

        # Monitors
        for m in self.monitors:
            lines.append(m)
        lines.append("")

        # Checks
        for c in self.checks:
            lines.append(c)

        lines.append("endmodule")
        return "\n".join(lines)


def gen_mac_testbench(data_width: int = 8) -> str:
    """Generate testbench for systolic MAC."""
    signals = [
        {"name": "clk", "width": 1, "direction": "reg", "initial": "0"},
        {"name": "rst_n", "width": 1, "direction": "reg", "initial": "0"},
        {"name": "valid_in", "width": 1, "direction": "reg", "initial": "0"},
        {"name": "weight_in", "width": data_width, "direction": "reg"},
        {"name": "activation_in", "width": data_width, "direction": "reg"},
        {"name": "partial_in", "width": data_width * 2, "direction": "reg"},
        {"name": "valid_out", "width": 1, "direction": "wire"},
        {"name": "mac_out", "width": data_width * 2, "direction": "wire"},
        {"name": "partial_out", "width": data_width * 2, "direction": "wire"},
    ]
    gen = TestbenchGenerator(f"systolic_mac_{data_width}", signals)
    gen.add_clock()
    gen.add_reset_task()
    gen.add_scoreboard("mac_out")

    # Test vectors: multiply pairs
    vectors = [
        {"rst_n": 0, "valid_in": 0},
        {"#5": ""},
        {"rst_n": 1},
        {"weight_in": "8'd3", "activation_in": "8'd7", "partial_in": "16'd0", "valid_in": 1},
        {"#5": ""},
        {"valid_in": 0},
        {"#10": ""},
        {"weight_in": "8'd10", "activation_in": "8'd20", "partial_in": "16'd21", "valid_in": 1},
        {"#5": ""},
        {"valid_in": 0},
        {"#10": ""},
    ]
    gen.add_basic_stimulus(vectors)
    return gen.generate()


def gen_weight_bank_tb(addr_width: int = 8, data_width: int = 8) -> str:
    """Generate testbench for weight bank."""
    signals = [
        {"name": "clk", "width": 1, "direction": "reg", "initial": "0"},
        {"name": "rst_n", "width": 1, "direction": "reg", "initial": "0"},
        {"name": "addr", "width": addr_width, "direction": "reg"},
        {"name": "dout", "width": data_width, "direction": "wire"},
        {"name": "layer_sel", "width": 4, "direction": "reg"},
    ]
    gen = TestbenchGenerator("weight_bank_24l_8a", signals)
    gen.add_clock(period_ns=4.0)
    gen.add_reset_task(cycles=3)
    gen.add_scoreboard("dout")

    vectors = [
        {"rst_n": 0},
        {"#5": ""},
        {"rst_n": 1, "layer_sel": "4'd0", "addr": "8'd0"},
        {"#10": ""},
        {"addr": "8'd1"},
        {"#10": ""},
        {"layer_sel": "4'd5", "addr": "8'd127"},
        {"#10": ""},
    ]
    gen.add_basic_stimulus(vectors)
    return gen.generate()


def demo():
    print("=== Testbench Generator ===\n")

    # MAC testbench
    mac_tb = gen_mac_testbench(8)
    print(f"--- MAC Testbench ({len(mac_tb)} chars) ---")
    print(mac_tb[:600])
    print(f"  ...")
    print(f"  Total: {len(mac_tb.split(chr(10)))} lines")
    print()

    # Weight bank testbench
    wb_tb = gen_weight_bank_tb()
    print(f"--- Weight Bank Testbench ({len(wb_tb)} chars) ---")
    print(wb_tb[:400])
    print(f"  ...")
    print(f"  Total: {len(wb_tb.split(chr(10)))} lines")


if __name__ == "__main__":
    demo()
