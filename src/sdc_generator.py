#!/usr/bin/env python3
"""SDC (Synopsys Design Constraints) generator for mask-locked chips.

Generates timing, power, and area constraints for synthesis,
place-and-route, and signoff.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ClockDomain:
    name: str
    period_ns: float
    waveform: str  # e.g. "0 2.5" for 50% duty
    source: str    # port name
    uncertainty_ns: float = 0
    latency_ns: float = 0
    generated: bool = False


@dataclass
class IOConstraint:
    port: str
    direction: str  # input, output
    delay_ns: float
    clock: str
    driving_cell: str = ""
    load_ff: float = 0


class SDCGenerator:
    """Generate SDC constraints for synthesis and PnR."""

    def __init__(self, design_name: str = "frozen_inference",
                 process_nm: int = 28):
        self.design = design_name
        self.nm = process_nm
        self.lines: List[str] = []
        self.clocks: Dict[str, ClockDomain] = {}
        self.ios: List[IOConstraint] = []

        # Process-specific defaults
        self.wire_load = "28nm_wl"
        self.operating_conditions = "typical"

    def set_design(self, name: str):
        self.design = name
        self.lines.append(f"set_design {name}")

    def add_clock(self, name: str, period_ns: float, source: str,
                  duty: float = 0.5, uncertainty: float = 0.1):
        half = period_ns * duty
        waveform = f"0 {half:.3f}"
        clk = ClockDomain(name, period_ns, waveform, source, uncertainty)
        self.clocks[name] = clk
        self.lines.append(f"create_clock -name {name} -period {period_ns:.3f} "
                         f"-waveform {{{waveform}}} {source}")
        if uncertainty > 0:
            self.lines.append(f"set_clock_uncertainty -setup "
                             f"{uncertainty:.3f} [get_clocks {name}]")

    def add_generated_clock(self, name: str, period_ns: float,
                             source: str, divide_by: int,
                             master_clk: str):
        self.add_clock(name, period_ns, source)
        self.clocks[name].generated = True
        self.lines.append(f"create_generated_clock -name {name} -master_clock "
                         f"[get_clocks {master_clk}] -divide_by {divide_by} "
                         f"-source {source} {source}")

    def add_io_delay(self, port: str, direction: str, delay_ns: float,
                     clock: str, clock_fall: bool = False,
                     max_min: str = "max"):
        self.ios.append(IOConstraint(port, direction, delay_ns, clock))
        edge = "-clock_fall " if clock_fall else ""
        self.lines.append(f"set_{direction}_delay -{max_min} {edge}"
                         f"-clock [get_clocks {clock}] {delay_ns:.3f} {port}")

    def set_driving_cell(self, ports: str, cell: str = "BUF_X4"):
        self.lines.append(f"set_driving_cell -lib_cell {cell} {ports}")

    def set_load(self, ports: str, load_ff: float = 5.0):
        self.lines.append(f"set_load {load_ff:.1f} {ports}")

    def set_input_transition(self, ports: str, transition_ns: float = 0.1):
        self.lines.append(f"set_input_transition {transition_ns:.3f} {ports}")

    def set_max_delay(self, from_pins: str, to_pins: str,
                      delay_ns: float, through: str = ""):
        cmd = f"set_max_delay {delay_ns:.3f} -from {from_pins} -to {to_pins}"
        if through:
            cmd += f" -through {through}"
        self.lines.append(cmd)

    def set_false_path(self, from_pins: str = "", to_pins: str = "",
                       through_pins: str = ""):
        cmd = "set_false_path"
        if from_pins:
            cmd += f" -from {from_pins}"
        if to_pins:
            cmd += f" -to {to_pins}"
        if through_pins:
            cmd += f" -through {through_pins}"
        self.lines.append(cmd)

    def set_multicycle_path(self, setup: int = 2, hold: int = 0,
                            from_pins: str = "", to_pins: str = ""):
        cmd = f"set_multicycle_path -setup {setup}"
        if from_pins:
            cmd += f" -from {from_pins}"
        if to_pins:
            cmd += f" -to {to_pins}"
        self.lines.append(cmd)

        hold_cycles = max(0, setup - 2) if hold == 0 else hold
        if hold_cycles > 0:
            hold_cmd = f"set_multicycle_path -hold {hold_cycles}"
            if from_pins:
                hold_cmd += f" -from {from_pins}"
            if to_pins:
                hold_cmd += f" -to {to_pins}"
            self.lines.append(hold_cmd)

    def set_max_area(self, area_um2: float = 0):
        self.lines.append(f"set_max_area {area_um2:.0f}")

    def set_max_power(self, power_mw: float = 0):
        self.lines.append(f"set_max_dynamic_power {power_mw:.1f}")

    def set_max_fanout(self, fanout: int = 32):
        self.lines.append(f"set_max_fanout {fanout}")
        self.lines.append(f"set_max_transition 0.5 [current_design]")
        self.lines.append(f"set_max_capacitance 0.5 [current_design]")

    def generate_reset_constraints(self, rst_port: str = "rst_n",
                                    is_async: bool = True):
        if is_async:
            self.lines.append(f"set_false_path -from {rst_port}")
            self.lines.append(f"set_disable_timing [get_cells -hier *reset_sync*]")

    def generate(self) -> str:
        """Return full SDC file."""
        header = [
            f"# SDC constraints for {self.design}",
            f"# Process: {self.nm}nm",
            f"# Generated by frozen-intelligence SDC generator",
            "",
        ]
        return "\n".join(header + self.lines + [""])


def demo():
    print("=== SDC Constraint Generator ===\n")

    sdc = SDCGenerator("frozen_scout_1b", 28)

    # Clocks
    sdc.add_clock("clk", 2.0, "clk", duty=0.5, uncertainty=0.1)
    sdc.add_generated_clock("clk_div2", 4.0, "clk_div2_reg/Q", 2, "clk")

    # I/O
    sdc.add_io_delay("data_in[*]", "input", 0.5, "clk")
    sdc.add_io_delay("valid_in", "input", 0.3, "clk")
    sdc.add_io_delay("result[*]", "output", 0.5, "clk")
    sdc.add_io_delay("valid_out", "output", 0.3, "clk")

    # Driving and loading
    sdc.set_driving_cell("[all_inputs]", "BUF_X4")
    sdc.set_load("[all_outputs]", 5.0)
    sdc.set_input_transition("[all_inputs]", 0.15)

    # Timing exceptions
    sdc.set_false_path(from_pins="rst_n")
    sdc.set_false_path(from_pins="scan_en")
    sdc.set_multicycle_path(setup=3, from_pins="weight_data[*]",
                           to_pins="mac_array[*]")

    # Design constraints
    sdc.set_max_fanout(24)
    sdc.set_max_area(0)  # no area constraint
    sdc.generate_reset_constraints()

    # Output
    print("--- Generated SDC ---")
    sdc_text = sdc.generate()
    print(sdc_text)

    # Summary
    print("--- Summary ---")
    print(f"  Clocks: {len(sdc.clocks)}")
    print(f"  I/O constraints: {len(sdc.ios)}")
    print(f"  Total lines: {len(sdc.lines)}")

    # Frequency info
    for name, clk in sdc.clocks.items():
        freq = 1000 / clk.period_ns
        print(f"  {name}: {clk.period_ns}ns ({freq:.0f}MHz)")


if __name__ == "__main__":
    demo()
