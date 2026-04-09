#!/usr/bin/env python3
"""RTL optimizer for mask-locked chip designs.

Constant folding, common subexpression elimination,
dead code elimination, and operation chaining.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set


@dataclass
class RTLInstr:
    op: str           # ADD, SUB, MUL, SHL, SHR, AND, OR, XOR, MUX, REG
    dst: str          # destination
    src: List[str]    # source operands
    width: int = 32
    constant: int = None  # if result is a known constant


@dataclass
class OptReport:
    original_instrs: int
    optimized_instrs: int
    constants_folded: int
    cse_eliminated: int
    dead_code_removed: int
    reduction_pct: float


class RTLOptimizer:
    """Optimize RTL instruction sequences."""

    def __init__(self):
        self.instructions: List[RTLInstr] = []
        self.constants: Dict[str, int] = {}  # signal -> known value
        self.computed: Dict[str, int] = {}   # expression hash -> result signal

    def parse_verilog_assign(self, line: str) -> Optional[RTLInstr]:
        """Parse simple Verilog assign: assign dst = src1 OP src2;"""
        line = line.strip().rstrip(";")
        m = re.match(r"assign\s+(\w+)\s*=\s*(.+)", line)
        if not m:
            return None
        dst = m.group(1)
        expr = m.group(2).strip()

        # Try binary ops
        ops = [("+", "ADD"), ("-", "SUB"), ("*", "MUL"),
               ("&", "AND"), ("|", "OR"), ("^", "XOR"),
               (">>", "SHR"), ("<<", "SHL")]
        for sym, op in ops:
            parts = expr.split(sym)
            if len(parts) == 2:
                return RTLInstr(op, dst, [parts[0].strip(), parts[1].strip()])

        # Simple wire
        return RTLInstr("WIRE", dst, [expr])

    def constant_fold(self) -> int:
        """Fold constant expressions. Returns number folded."""
        folded = 0
        for instr in self.instructions:
            # Check if all sources are known constants
            src_vals = []
            all_const = True
            for s in instr.src:
                val = self.constants.get(s)
                if val is not None:
                    src_vals.append(val)
                elif s.isdigit():
                    src_vals.append(int(s))
                elif s.startswith("0x"):
                    src_vals.append(int(s, 16))
                else:
                    all_const = False
                    break

            if all_const and len(src_vals) == 2:
                a, b = src_vals
                result = None
                if instr.op == "ADD": result = a + b
                elif instr.op == "SUB": result = a - b
                elif instr.op == "MUL": result = a * b
                elif instr.op == "AND": result = a & b
                elif instr.op == "OR": result = a | b
                elif instr.op == "XOR": result = a ^ b
                elif instr.op == "SHR": result = a >> b
                elif instr.op == "SHL": result = a << b

                if result is not None:
                    mask = (1 << instr.width) - 1
                    instr.constant = result & mask
                    self.constants[instr.dst] = instr.constant
                    instr.src = [str(instr.constant)]  # replace with constant
                    folded += 1
            elif all_const and len(src_vals) == 1:
                instr.constant = src_vals[0]
                self.constants[instr.dst] = instr.constant
                folded += 1

        return folded

    def cse(self) -> int:
        """Common subexpression elimination."""
        eliminated = 0
        expr_map = {}  # (op, tuple(sorted srcs)) -> existing dst

        new_instrs = []
        for instr in self.instructions:
            if instr.constant is not None:
                new_instrs.append(instr)
                continue

            key = (instr.op, tuple(instr.src))
            if key in expr_map:
                # Replace with existing result
                existing = expr_map[key]
                instr.op = "WIRE"
                instr.src = [existing]
                self.constants[instr.dst] = self.constants.get(existing)
                eliminated += 1
            else:
                expr_map[key] = instr.dst
            new_instrs.append(instr)

        self.instructions = new_instrs
        return eliminated

    def dead_code_elimination(self) -> int:
        """Remove instructions whose results are never used."""
        # Find all used signals
        used = set()
        for instr in self.instructions:
            for s in instr.src:
                used.add(s)

        # Remove dead (not used by anyone and not a primary output)
        # Assume signals starting with 'out_' are outputs
        new_instrs = []
        removed = 0
        for instr in self.instructions:
            if instr.dst not in used and not instr.dst.startswith("out_"):
                if instr.constant is None:  # don't remove constants we computed
                    removed += 1
                    continue
            new_instrs.append(instr)

        self.instructions = new_instrs
        return removed

    def optimize(self, instructions: List[RTLInstr]) -> Tuple[List[RTLInstr], OptReport]:
        """Run full optimization pipeline."""
        self.instructions = list(instructions)
        original = len(self.instructions)

        # Multi-pass
        total_folded = 0
        total_cse = 0
        total_dce = 0

        for _ in range(3):  # iterate until stable
            folded = self.constant_fold()
            total_folded += folded
            cse = self.cse()
            total_cse += cse
            dce = self.dead_code_elimination()
            total_dce += dce

        optimized = len(self.instructions)
        reduction = (1 - optimized / original) * 100 if original > 0 else 0

        return self.instructions, OptReport(
            original_instrs=original,
            optimized_instrs=optimized,
            constants_folded=total_folded,
            cse_eliminated=total_cse,
            dead_code_removed=total_dce,
            reduction_pct=round(reduction, 1),
        )


def demo():
    print("=== RTL Optimizer ===\n")

    opt = RTLOptimizer()

    # Test instructions (typical MAC unit data path)
    instrs = [
        RTLInstr("WIRE", "bias", ["1024"]),
        RTLInstr("SHL", "a_scaled", ["input_a", "2"]),
        RTLInstr("MUL", "a_x_b", ["a_scaled", "weight_b"]),
        RTLInstr("MUL", "c_x_d", ["input_c", "weight_d"]),
        RTLInstr("ADD", "partial1", ["a_x_b", "c_x_d"]),
        RTLInstr("ADD", "out_result", ["partial1", "bias"]),
        # Dead code
        RTLInstr("MUL", "unused1", ["input_a", "input_c"]),
        RTLInstr("ADD", "unused2", ["bias", "1024"]),
        # CSE opportunity
        RTLInstr("MUL", "a_x_b_2", ["a_scaled", "weight_b"]),
        RTLInstr("ADD", "partial2", ["a_x_b_2", "bias"]),
        # Constant fold
        RTLInstr("ADD", "const_sum", ["1024", "256"]),
        RTLInstr("MUL", "const_mul", ["const_sum", "2"]),
        RTLInstr("ADD", "out_const", ["const_mul", "128"]),
    ]

    # Seed known constants
    opt.constants = {"1024": 1024, "256": 256, "2": 2, "128": 128}

    print("--- Original Instructions ---")
    for i in instrs:
        src_str = " ".join(i.src)
        const = f" = {i.constant}" if i.constant is not None else ""
        print(f"  {i.op:6s} {i.dst:15s} <- {src_str}{const}")
    print(f"  Total: {len(instrs)} instructions")
    print()

    # Optimize
    optimized, report = opt.optimize(instrs)

    print(f"--- Optimization Report ---")
    print(f"  Original: {report.original_instrs} instructions")
    print(f"  Optimized: {report.optimized_instrs} instructions")
    print(f"  Constants folded: {report.constants_folded}")
    print(f"  CSE eliminated: {report.cse_eliminated}")
    print(f"  Dead code removed: {report.dead_code_removed}")
    print(f"  Reduction: {report.reduction_pct}%")
    print()

    print("--- Optimized Instructions ---")
    for i in optimized:
        src_str = " ".join(i.src)
        const = f" = {i.constant}" if i.constant is not None else ""
        print(f"  {i.op:6s} {i.dst:15s} <- {src_str}{const}")
    print()

    # Verilog parsing demo
    print("--- Verilog Parse Test ---")
    verilog_lines = [
        "assign a_scaled = input_a << 2;",
        "assign product = a_scaled * weight;",
        "assign result = product + 1024;",
        "assign shifted = result >> 4;",
    ]
    parsed = []
    for line in verilog_lines:
        instr = opt.parse_verilog_assign(line)
        if instr:
            parsed.append(instr)
            print(f"  {instr.op:6s} {instr.dst:15s} <- {' '.join(instr.src)}")


if __name__ == "__main__":
    demo()
