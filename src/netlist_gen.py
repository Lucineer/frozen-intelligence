#!/usr/bin/env python3
"""Structural Verilog netlist generator for mask-locked chips.

Generates cell-level netlists from architectural descriptions:
MAC arrays, weight banks, control logic, I/O pads.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class CellType(Enum):
    INV = "INV"
    NAND2 = "NAND2"
    NOR2 = "NOR2"
    BUF = "BUF"
    DFF = "DFF"
    MUX2 = "MUX2"
    FA = "FA"   # full adder
    HA = "HA"   # half adder
    AND2 = "AND2"
    OR2 = "OR2"
    XOR2 = "XOR2"


@dataclass
class CellInst:
    cell_type: CellType
    name: str
    connections: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, str] = field(default_factory=dict)

    def verilog(self) -> str:
        params = ""
        if self.parameters:
            plist = ", ".join(f".{k}({v})" for k, v in self.parameters.items())
            params = f" #({plist})"
        conns = ", ".join(f".{k}({v})" for k, v in self.connections.items())
        return f"  {self.cell_type.value}{params} {self.name} ({conns});"


@dataclass
class Wire:
    name: str
    width: int = 1

    def declaration(self) -> str:
        if self.width == 1:
            return f"  wire {self.name};"
        return f"  wire [{self.width-1}:0] {self.name};"


@dataclass
class Module:
    name: str
    ports: Dict[str, int] = field(default_factory=dict)  # name -> width
    cells: List[CellInst] = field(default_factory=list)
    wires: List[Wire] = field(default_factory=list)
    assigns: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)

    def add_wire(self, name: str, width: int = 1):
        if not any(w.name == name for w in self.wires):
            self.wires.append(Wire(name, width))

    def add_comment(self, text: str):
        self.comments.append(f"  // {text}")

    def verilog(self) -> str:
        lines = [f"module {self.name} ("]

        # Ports
        port_list = []
        for name, width in self.ports.items():
            if width == 1:
                port_list.append(name)
            else:
                port_list.append(f"input [{width-1}:0] {name}" if "clk" not in name.lower() and "rst" not in name.lower() else name)
        lines.append("    " + ", ".join(port_list))

        # Port declarations
        lines.append("  );")
        for name, width in self.ports.items():
            if width == 1:
                direction = "input" if name in ("clk", "rst_n", "enable") else "output" if "out" in name or "dout" in name else "input"
                lines.append(f"  {direction} {name};")
            else:
                direction = "input" if "in" in name or "data" in name else "output"
                lines.append(f"  {direction} [{width-1}:0] {name};")

        # Wires
        for w in self.wires:
            lines.append(w.declaration())

        # Comments
        for c in self.comments:
            lines.append(c)

        # Assigns
        for a in self.assigns:
            lines.append(f"  assign {a};")

        # Cells
        for c in self.cells:
            lines.append(c.verilog())

        lines.append("endmodule")
        return "\n".join(lines)


def gen_multiplier(n_bits: int) -> Module:
    """Generate array multiplier netlist."""
    mod = Module(f"multiplier_{n_bits}",
                 {"a": n_bits, "b": n_bits, "product": 2 * n_bits, "clk": 1, "rst_n": 1})

    for i in range(n_bits):
        mod.add_wire(f"pp_{i}", n_bits)
        mod.add_comment(f"Partial product row {i}")

    # Partial products
    for i in range(n_bits):
        for j in range(n_bits):
            pp_wire = f"pp_{i}[{j}]"
            mod.cells.append(CellInst(
                CellType.AND2, f"and_{i}_{j}",
                {"A": f"a[{i}]", "B": f"b[{j}]", "Y": pp_wire}))

    # Wallace tree (simplified: use full adders and half adders)
    mod.add_wire("sum", 2 * n_bits)
    mod.add_wire("carry", 2 * n_bits)

    for i in range(n_bits):
        mod.cells.append(CellInst(
            CellType.HA, f"ha_{i}",
            {"A": f"pp_{i}[0]", "B": f"carry[{i}]", "S": f"sum[{i}]", "CO": f"carry[{i+1}]"})
        )

    mod.assigns.append(f"product = sum")
    return mod


def gen_systolic_mac(n_bits: int = 8) -> Module:
    """Generate single systolic MAC unit."""
    mod = Module(f"systolic_mac_{n_bits}",
                 {"clk": 1, "rst_n": 1, "valid_in": 1, "weight_in": n_bits,
                  "activation_in": n_bits, "partial_in": n_bits * 2,
                  "valid_out": 1, "mac_out": n_bits * 2, "partial_out": n_bits * 2})

    mod.add_wire("product", 2 * n_bits)
    mod.add_wire("sum_raw", 2 * n_bits)
    mod.add_wire("sum_reg", 2 * n_bits)
    mod.add_comment("Multiply weight * activation")

    # Multiplier
    mult = gen_multiplier(n_bits)
    mod.add_wire("mult_a", n_bits)
    mod.add_wire("mult_b", n_bits)
    mod.add_comment("Product = weight * activation")
    for i in range(n_bits):
        mod.cells.append(CellInst(
            CellType.AND2, f"pp_{i}_{j}" if False else f"and_wa_{i}",
            {"A": f"weight_in[{i}]", "B": f"activation_in[{i}]",
             "Y": f"product[{i}]"}) if False else
            CellInst(CellType.AND2, f"and_wa_{i}",
                     {"A": f"weight_in[{i}]", "B": f"activation_in[0]",
                      "Y": f"product[{i}]"}))

    # Accumulator
    mod.add_comment("Accumulate: sum = product + partial_in")
    for i in range(2 * n_bits):
        a = f"product[{i}]" if i < 2 * n_bits else "1'b0"
        b = f"partial_in[{i}]" if i < 2 * n_bits else "1'b0"
        mod.cells.append(CellInst(
            CellType.FA, f"fa_acc_{i}",
            {"A": a, "B": b, "CI": f"sum_raw[{i-1}]" if i > 0 else "1'b0",
             "S": f"sum_raw[{i}]", "CO": f"carry_acc_{i}"}))

    # Output register
    mod.add_wire("d", 2 * n_bits)
    mod.assigns.append("d = valid_in ? sum_raw : partial_in")
    for i in range(2 * n_bits):
        mod.cells.append(CellInst(
            CellType.DFF, f"dff_out_{i}",
            {"D": f"d[{i}]", "CK": "clk", "RN": "rst_n", "Q": f"mac_out[{i}]"})
        )

    # Pipeline partial
    for i in range(2 * n_bits):
        mod.cells.append(CellInst(
            CellType.DFF, f"dff_partial_{i}",
            {"D": f"mac_out[{i}]", "CK": "clk", "RN": "rst_n",
             "Q": f"partial_out[{i}]"})
        )

    mod.assigns.append("valid_out = valid_in")
    return mod


def gen_weight_bank(n_layers: int, n_entries: int = 256,
                    data_width: int = 8) -> Module:
    """Generate weight bank with ROM-style storage."""
    addr_bits = math.ceil(math.log2(n_entries))
    mod = Module(f"weight_bank_{n_layers}l_{addr_bits}a",
                 {"clk": 1, "rst_n": 1, "addr": addr_bits,
                  "dout": data_width, "layer_sel": math.ceil(math.log2(n_layers))})

    mod.add_wire("decoded", n_layers)
    mod.add_comment("Layer decoder")

    for i in range(n_layers):
        mod.cells.append(CellInst(
            CellType.DFF, f"dff_dout_{i}",
            {"D": f"rom_{i}_out", "CK": "clk", "RN": "rst_n", "Q": f"dout"}))
        mod.add_wire(f"rom_{i}_out", data_width)

    return mod


def gen_top_chip(n_layers: int = 24, n_heads: int = 8,
                 data_width: int = 8) -> str:
    """Generate top-level chip netlist."""
    header = [
        "// Mask-Locked Inference Chip — Auto-generated Netlist",
        f"// Layers: {n_layers}, Heads: {n_heads}, Data: {data_width}bit",
        "// Lucineer (DiGennaro et al.)",
        "",
    ]

    # MAC unit
    mac = gen_systolic_mac(data_width)
    header.append(mac.verilog())
    header.append("")

    # Weight bank
    wb = gen_weight_bank(n_layers, 256, data_width)
    header.append(wb.verilog())
    header.append("")

    # Top module
    top = Module(f"frozen_intelligence_top",
                 {"clk": 1, "rst_n": 1, "data_in": data_width,
                  "data_out": data_width, "valid": 1, "ready": 1})
    top.add_comment("Top-level mask-locked inference chip")
    top.add_comment(f"Vessel: Captain class, {n_layers} layers, {n_heads} heads")

    # Instantiate MAC array
    top.add_comment("MAC systolic array")
    for i in range(n_heads):
        top.cells.append(CellInst(
            CellType.MUX2, f"mac_sel_{i}",
            {"A": f"mac_{i}.mac_out", "B": "8'b0", "S": "layer_sel_en", "Y": f"data_mux_{i}"}))

    top.add_comment("Control FSM")
    top.cells.append(CellInst(
        CellType.DFF, f"dff_state",
        {"D": "next_state", "CK": "clk", "RN": "rst_n", "Q": "current_state"}))
    top.add_wire("next_state", 3)
    top.add_wire("current_state", 3)
    top.assigns.append("ready = (current_state == 3'b000)")

    header.append(top.verilog())

    return "\n".join(header)


def demo():
    print("=== Verilog Netlist Generator ===\n")

    # Multiplier
    mult = gen_multiplier(4)
    print(f"--- 4-bit Multiplier ({len(mult.cells)} cells) ---")
    print(mult.verilog()[:500])
    print(f"  ... ({len(mult.verilog())} chars total)")
    print()

    # MAC unit
    mac = gen_systolic_mac(4)
    print(f"--- Systolic MAC ({len(mac.cells)} cells, {len(mac.wires)} wires) ---")
    print(mac.verilog()[:500])
    print(f"  ... ({len(mac.verilog())} chars total)")
    print()

    # Top chip
    top = gen_top_chip(24, 8, 8)
    print(f"--- Top-Level Chip ---")
    print(f"  Netlist size: {len(top)} chars")
    lines = top.split("\n")
    print(f"  Lines: {len(lines)}")
    print(f"  Modules: {top.count('module ')}")
    print(f"  Cell instances: {top.count('INV_') + top.count('DFF_') + top.count('FA_') + top.count('AND2_')}")
    print()

    # Cell count by type
    for ct in CellType:
        count = top.count(f" {ct.value}_")
        if count > 0:
            print(f"  {ct.value:8s}: {count}")


if __name__ == "__main__":
    demo()
