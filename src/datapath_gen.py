#!/usr/bin/env python3
"""RTL datapath generator for mask-locked inference chips.

Auto-generates Verilog for MAC units, adder trees, barrel shifters,
and complete inference datapaths with pipeline registers.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


class VerilogGen:
    """Verilog code generator utilities."""

    @staticmethod
    def header(name: str, width: int = 32) -> str:
        return f"module {name} (\n    input clk,\n    input rst_n,\n    output reg [{width-1}:0] result\n);\n"

    @staticmethod
    def footer() -> str:
        return "endmodule"

    @staticmethod
    def wire(name: str, width: int = 32) -> str:
        return f"wire [{width-1}:0] {name};\n"

    @staticmethod
    def reg(name: str, width: int = 32) -> str:
        return f"reg [{width-1}:0] {name};\n"

    @staticmethod
    def assign(dst: str, expr: str) -> str:
        return f"assign {dst} = {expr};\n"

    @staticmethod
    def always_ff(clk: str, rst: str, body: str, async_rst: bool = True) -> str:
        if async_rst:
            return f"always @(posedge clk or negedge {rst}) begin\n    if (!{rst}) begin\n        {body}\n    end\nend\n"
        return f"always @(posedge clk) begin\n    {body}\nend\n"

    @staticmethod
    def pipeline_reg(name: str, width: int, src: str, clk: str = "clk",
                     rst: str = "rst_n") -> str:
        return (f"always @(posedge clk or negedge {rst}) begin\n"
                f"    if (!{rst})\n        {name} <= 0;\n"
                f"    else\n        {name} <= {src};\n"
                f"end\n")


class DatapathGenerator:
    """Generate RTL datapaths for inference."""

    def __init__(self, data_width: int = 16, pipe_stages: int = 3):
        self.W = data_width
        self.pipe = pipe_stages
        self.vg = VerilogGen()

    def generate_mac_unit(self, name: str = "mac_unit") -> str:
        """Generate pipelined multiply-accumulate unit."""
        W = self.W
        P = W * 2  # product width
        lines = []

        lines.append(f"module {name} (")
        lines.append(f"    input clk,")
        lines.append(f"    input rst_n,")
        lines.append(f"    input [{W-1}:0] a,")
        lines.append(f"    input [{W-1}:0] b,")
        lines.append(f"    input [{P-1}:0] acc_in,")
        lines.append(f"    input valid_in,")
        lines.append(f"    output reg [{P-1}:0] acc_out,")
        lines.append(f"    output reg valid_out")
        lines.append(f");")

        # Pipeline stage 1: multiply
        lines.append(f"    reg [{P-1}:0] product;")
        lines.append(f"    reg valid_s1;")
        lines.append(self.vg.pipeline_reg("product", P, f"a * b"))
        lines.append(self.vg.pipeline_reg("valid_s1", 1, "valid_in"))

        # Pipeline stage 2: accumulate
        lines.append(f"    reg [{P-1}:0] sum;")
        lines.append(f"    reg valid_s2;")
        lines.append(self.vg.pipeline_reg("sum", P, "product + acc_in"))
        lines.append(self.vg.pipeline_reg("valid_s2", 1, "valid_s1"))

        # Pipeline stage 3: output
        lines.append(f"    always @(posedge clk or negedge rst_n) begin")
        lines.append(f"        if (!rst_n) begin")
        lines.append(f"            acc_out <= 0;")
        lines.append(f"            valid_out <= 0;")
        lines.append(f"        end else begin")
        lines.append(f"            acc_out <= sum;")
        lines.append(f"            valid_out <= valid_s2;")
        lines.append(f"        end")
        lines.append(f"    end")

        lines.append(f"endmodule")
        return "\n".join(lines)

    def generate_adder_tree(self, n_inputs: int, name: str = "adder_tree") -> str:
        """Generate balanced adder tree."""
        W = self.W
        lines = []

        lines.append(f"module {name} (")
        lines.append(f"    input clk,")
        lines.append(f"    input rst_n,")
        lines.append(f"    input [{W-1}:0] data [{n_inputs-1}:0],")
        lines.append(f"    input valid_in,")
        lines.append(f"    output reg [{W-1}:0] sum_out,")
        lines.append(f"    output reg valid_out")
        lines.append(f");")

        # Build tree levels
        current = [f"data[{i}]" for i in range(n_inputs)]
        stage = 0

        while len(current) > 1:
            next_level = []
            for i in range(0, len(current) - 1, 2):
                w = W
                lines.append(f"    reg [{w-1}:0] add_s{stage}_{i//2};")
                lines.append(f"    wire [{w-1}:0] add_s{stage}_{i//2}_comb;")
                lines.append(f"    assign add_s{stage}_{i//2}_comb = {current[i]} + {current[i+1]};")
                lines.append(f"    always @(posedge clk) add_s{stage}_{i//2} <= add_s{stage}_{i//2}_comb;")
                next_level.append(f"add_s{stage}_{i//2}")

            if len(current) % 2 == 1:
                next_level.append(current[-1])

            current = next_level
            stage += 1

        # Output
        lines.append(f"    reg valid_pipe [{stage-1}:0];")
        lines.append(f"    always @(posedge clk or negedge rst_n) begin")
        lines.append(f"        if (!rst_n) begin")
        lines.append(f"            valid_pipe <= 0;")
        lines.append(f"            valid_out <= 0;")
        lines.append(f"        end else begin")
        lines.append(f"            valid_pipe[0] <= valid_in;")
        lines.append(f"            for (int i = 1; i < {stage}; i++) valid_pipe[i] <= valid_pipe[i-1];")
        lines.append(f"            valid_out <= valid_pipe[{stage-1}];")
        lines.append(f"        end")
        lines.append(f"    end")
        lines.append(f"    assign sum_out = {current[0]};")

        lines.append(f"endmodule")
        return "\n".join(lines)

    def generate_barrel_shifter(self, name: str = "barrel_shifter") -> str:
        """Generate logarithmic barrel shifter."""
        W = self.W
        log_w = int(math.log2(W))
        lines = []

        lines.append(f"module {name} (")
        lines.append(f"    input [{W-1}:0] data_in,")
        lines.append(f"    input [{log_w-1}:0] shift_amt,")
        lines.append(f"    input shift_right,")
        lines.append(f"    output reg [{W-1}:0] data_out")
        lines.append(f");")

        # Logarithmic shift stages
        prev = f"data_in"
        for i in range(log_w):
            amt = 1 << i
            cur = f"stage_{i}"
            lines.append(f"    wire [{W-1}:0] {cur};")
            lines.append(f"    assign {cur} = shift_right ?")
            lines.append(f"        {{ {amt}'b0, {prev}[{W-1}:{amt}] }} :")
            lines.append(f"        {{ {prev}[{W-1}-{amt}:0], {amt}'b0 }};")
            prev = cur

        lines.append(f"    always @(*) data_out = {prev};")
        lines.append(f"endmodule")
        return "\n".join(lines)

    def generate_activation(self, act_type: str = "relu",
                            name: str = "activation") -> str:
        """Generate activation function."""
        W = self.W
        lines = []

        lines.append(f"module {name} (")
        lines.append(f"    input [{W-1}:0] data_in,")
        lines.append(f"    output reg [{W-1}:0] data_out")
        lines.append(f");")

        if act_type == "relu":
            lines.append(f"    always @(*) begin")
            lines.append(f"        data_out = (data_in[W-1]) ? 0 : data_in;")
            lines.append(f"    end")
        elif act_type == "gelu":
            lines.append(f"    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))")
            lines.append(f"    always @(*) begin")
            lines.append(f"        // Simplified: use ReLU for mask-locked (exact GELU needs LUT)")
            lines.append(f"        data_out = (data_in[{W-1}]) ? 0 : data_in;")
            lines.append(f"    end")
        elif act_type == "sigmoid":
            lines.append(f"    // Sigmoid: for mask-locked, use piecewise linear approx")
            lines.append(f"    wire signed [{W-1}:0] s_in = $signed(data_in);")
            lines.append(f"    always @(*) begin")
            lines.append(f"        if (s_in > {1<<(W-2)}) data_out = {W}'d{(1<<(W))-1};")
            lines.append(f"        else if (s_in < -{1<<(W-2)}) data_out = 0;")
            lines.append(f"        else data_out = {{1'b0, s_in[{W-2}:0]}} + {1<<(W-2)};")
            lines.append(f"    end")

        lines.append(f"endmodule")
        return "\n".join(lines)

    def generate_weight_loader(self, n_layers: int = 4, W: int = 16,
                               name: str = "weight_loader") -> str:
        """Generate weight streaming interface."""
        lines = []

        lines.append(f"module {name} (")
        lines.append(f"    input clk,")
        lines.append(f"    input rst_n,")
        lines.append(f"    input [{W-1}:0] weight_data,")
        lines.append(f"    input weight_valid,")
        lines.append(f"    input weight_sop,  // start of packet")
        lines.append(f"    input weight_eop,  // end of packet")
        lines.append(f"    output reg weight_ready,")
        lines.append(f"    output reg [{15}:0] layer_id,")
        lines.append(f"    output reg [{15}:0] weight_count,")
        lines.append(f"    output reg load_complete")
        lines.append(f");")

        lines.append(f"    always @(posedge clk or negedge rst_n) begin")
        lines.append(f"        if (!rst_n) begin")
        lines.append(f"            weight_ready <= 1;")
        lines.append(f"            layer_id <= 0;")
        lines.append(f"            weight_count <= 0;")
        lines.append(f"            load_complete <= 0;")
        lines.append(f"        end else if (weight_valid && weight_ready) begin")
        lines.append(f"            weight_count <= weight_count + 1;")
        lines.append(f"            if (weight_eop) begin")
        lines.append(f"                layer_id <= layer_id + 1;")
        lines.append(f"                weight_count <= 0;")
        lines.append(f"                if (layer_id >= {n_layers - 1})")
        lines.append(f"                    load_complete <= 1;")
        lines.append(f"            end")
        lines.append(f"        end")
        lines.append(f"    end")

        lines.append(f"endmodule")
        return "\n".join(lines)


def demo():
    print("=== RTL Datapath Generator ===\n")

    gen = DatapathGenerator(16, 3)
    print(f"Config: {gen.W}-bit data, {gen.pipe} pipeline stages")
    print()

    # MAC unit
    print("--- MAC Unit ---")
    mac = gen.generate_mac_unit("inference_mac")
    print(mac[:500])
    print(f"  ... ({len(mac)} chars total)")
    print()

    # Adder tree
    print("--- 8-Input Adder Tree ---")
    tree = gen.generate_adder_tree(8, "mac_adder_tree")
    stages = tree.count("stage_0")
    print(f"  Total chars: {len(tree)}")
    print(f"  Pipeline stages: {stages}")
    print()

    # Barrel shifter
    print("--- Barrel Shifter ---")
    shifter = gen.generate_barrel_shifter("data_shifter")
    print(shifter)
    print()

    # Activation
    print("--- Activation Functions ---")
    for act in ["relu", "sigmoid", "gelu"]:
        mod = gen.generate_activation(act, f"{act}_act")
        print(f"  {act}: {len(mod)} chars")

    # Weight loader
    print()
    print("--- Weight Loader ---")
    loader = gen.generate_weight_loader(4, 16, "weight_stream")
    print(loader)
    print()

    # Top-level stats
    print("--- Summary ---")
    total = len(mac) + len(tree) + len(shifter) + len(loader)
    total += len(gen.generate_activation("relu"))
    print(f"  MAC unit: {len(mac)} chars")
    print(f"  Adder tree: {len(tree)} chars")
    print(f"  Barrel shifter: {len(shifter)} chars")
    print(f"  Total generated: {total} chars")


if __name__ == "__main__":
    demo()
