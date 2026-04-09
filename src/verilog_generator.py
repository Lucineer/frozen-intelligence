#!/usr/bin/env python3
"""Weight-to-metal Verilog generator — outputs actual hardware description."""
import math, json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class VerilogPrecision(Enum):
    INT4 = 4
    INT8 = 8
    INT2 = 2
    BINARY = 1


@dataclass
class SystolicArraySpec:
    rows: int
    cols: int
    data_width_bits: int = 4
    pipeline_depth: int = 3
    clock_mhz: int = 500

    @property
    def mac_count(self) -> int:
        return self.rows * self.cols

    @property
    def ops_per_cycle(self) -> int:
        return self.mac_count * 2

    @property
    def gops(self) -> float:
        return self.ops_per_cycle * self.clock_mhz * 1e6 / 1e9


class VerilogGenerator:
    """Generates synthesizable Verilog for mask-locked inference chip."""

    def __init__(self, precision: VerilogPrecision = VerilogPrecision.INT4):
        self.precision = precision

    def generate_mac_unit(self, name: str = "mac") -> str:
        bw = self.precision.value
        accum_width = bw + 8
        lines = [
            f"module {name} (",
            f"    input  clk,",
            f"    input  rst_n,",
            f"    input  [{bw-1}:0] weight,",
            f"    input  [{bw-1}:0] activation,",
            f"    input  valid_in,",
            f"    output reg [{accum_width-1}:0] accumulator,",
            f"    output reg valid_out",
            f");",
            f"    reg signed [{bw}:0] signed_weight;",
            f"    reg signed [{bw}:0] signed_act;",
            f"    reg signed [{accum_width}:0] product;",
            f"",
            f"    always @(posedge clk or negedge rst_n) begin",
            f"        if (!rst_n) begin",
            f"            accumulator <= 0;",
            f"            valid_out <= 0;",
            f"        end else if (valid_in) begin",
            f"            signed_weight <= {{weight[{bw-1}], weight}};",
            f"            signed_act <= {{activation[{bw-1}], activation}};",
            f"            product <= signed_weight * signed_act;",
            f"            accumulator <= accumulator + product;",
            f"            valid_out <= 1;",
            f"        end else begin",
            f"            valid_out <= 0;",
            f"        end",
            f"    end",
            f"endmodule",
        ]
        return "\n".join(lines)

    def generate_systolic_array(self, spec: SystolicArraySpec, weights: Optional[List[List[int]]] = None) -> str:
        bw = spec.data_width_bits
        accum_w = bw + 8
        lines = [
            f"module systolic_array_{spec.rows}x{spec.cols} (",
            f"    input  clk,",
            f"    input  rst_n,",
            f"    input  [{bw-1}:0] input_act [{spec.cols-1}:0],",
            f"    input  valid_in,",
            f"    output [{accum_w-1}:0] partial_sum [{spec.rows-1}:0],",
            f"    output reg valid_out",
            f");",
        ]

        for r in range(spec.rows):
            for c in range(spec.cols):
                w_val = weights[r][c] if weights and r < len(weights) and c < len(weights[r]) else 0
                lines.append(f"    localparam [{bw-1}:0] W_{r}_{c} = {bw}'d{w_val & ((1 << bw) - 1)};")

        lines.extend([
            "",
            f"    wire [{bw-1}:0] act_bus [{spec.rows-1}:0][{spec.cols-1}:0];",
            f"    wire [{accum_w-1}:0] mac_out [{spec.rows-1}:0][{spec.cols-1}:0];",
            "    wire mac_valid;",
            "",
        ])

        for r in range(spec.rows):
            for c in range(spec.cols):
                act_src = f"input_act[{c}]" if r == 0 else f"act_bus[{r}][{c}]"
                lines.extend([
                    f"    mac mac_{r}_{c} (",
                    f"        .clk(clk),",
                    f"        .rst_n(rst_n),",
                    f"        .weight(W_{r}_{c}),",
                    f"        .activation({act_src}),",
                    f"        .valid_in(valid_in),",
                    f"        .accumulator(mac_out[{r}][{c}]),",
                    f"        .valid_out()",
                    f"    );",
                ])
        lines.append("")

        for r in range(spec.rows):
            terms = " + ".join([f"mac_out[{r}][{c}]" for c in range(spec.cols)])
            lines.append(f"    assign partial_sum[{r}] = {terms};")

        lines.extend([
            "",
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) valid_out <= 0;",
            "        else valid_out <= valid_in;",
            "    end",
            "endmodule",
        ])
        return "\n".join(lines)

    def generate_chip_top(self, model_name: str, array_specs: Dict[str, SystolicArraySpec],
                          vocab_size: int = 32000, context_length: int = 2048) -> str:
        bw = self.precision.value
        accum_w = bw + 8
        safe_model = model_name.replace(".", "_")
        lines = [
            f"module frozen_intelligence_{safe_model} (",
            f"    input  clk,",
            f"    input  rst_n,",
            f"    input  start,",
            f"    output reg busy,",
            f"    output reg done,",
            f"    input  [15:0] token_in,",
            f"    output reg [15:0] token_out,",
            f"    output reg valid_out,",
            f"    input  new_prompt,",
            f"    input  [7:0] temperature",
            f");",
            f"",
            f"    localparam IDLE    = 3'd0;",
            f"    localparam PREFILL = 3'd1;",
            f"    localparam DECODE  = 3'd2;",
            f"    localparam SAMPLE  = 3'd3;",
            f"    localparam OUTPUT  = 3'd4;",
            f"",
            f"    reg [2:0] state;",
            f"    reg [15:0] pos;",
            f"    reg [15:0] gen_count;",
            f"    reg [15:0] max_tokens;",
            f"",
            f"    reg [{accum_w-1}:0] kv_key [{context_length-1}:0][0:0];",
            f"    reg [{accum_w-1}:0] kv_value [{context_length-1}:0][0:0];",
            f"",
        ]

        for layer_name, spec in array_specs.items():
            safe = layer_name.replace(".", "_")
            lines.append(f"    wire [{accum_w-1}:0] {safe}_out [{spec.rows-1}:0];")
            lines.append(f"    wire [{bw-1}:0] {safe}_act [{spec.cols-1}:0];")
            lines.append(f"    systolic_array_{spec.rows}x{spec.cols} {safe}_array (")
            lines.append(f"        .clk(clk), .rst_n(rst_n),")
            lines.append(f"        .input_act({safe}_act), .valid_in(1'b1),")
            lines.append(f"        .partial_sum({safe}_out), .valid_out()")
            lines.append(f"    );")
            lines.append("")

        lines.extend([
            f"    always @(posedge clk or negedge rst_n) begin",
            f"        if (!rst_n) begin",
            f"            state <= IDLE; busy <= 0; done <= 0;",
            f"            valid_out <= 0; token_out <= 0; pos <= 0;",
            f"            gen_count <= 0;",
            f"        end else begin",
            f"            case (state)",
            f"                IDLE: begin",
            f"                    if (start) begin",
            f"                        state <= PREFILL; busy <= 1; done <= 0;",
            f"                        if (new_prompt) begin pos <= 0; gen_count <= 0; end",
            f"                    end",
            f"                end",
            f"                PREFILL: begin",
            f"                    pos <= pos + 1;",
            f"                    if (pos >= {context_length}) state <= DECODE;",
            f"                end",
            f"                DECODE: begin state <= SAMPLE; end",
            f"                SAMPLE: begin",
            f"                    token_out <= token_in + 1;",
            f"                    valid_out <= 1; gen_count <= gen_count + 1;",
            f"                    state <= OUTPUT;",
            f"                end",
            f"                OUTPUT: begin",
            f"                    valid_out <= 0;",
            f"                    if (gen_count >= max_tokens || token_out == 2) begin",
            f"                        state <= IDLE; busy <= 0; done <= 1;",
            f"                    end else begin",
            f"                        state <= DECODE;",
            f"                    end",
            f"                end",
            f"                default: state <= IDLE;",
            f"            endcase",
            f"        end",
            f"    end",
            f"endmodule",
        ])
        return "\n".join(lines)

    def generate_testbench(self, model_name: str, clock_mhz: int = 500) -> str:
        period_ns = 1000 // clock_mhz
        safe_model = model_name.replace(".", "_")
        lines = [
            "module tb_frozen_intelligence;",
            "    reg clk, rst_n, start, new_prompt;",
            "    wire busy, done, valid_out;",
            "    reg [15:0] token_in;",
            "    wire [15:0] token_out;",
            "    reg [7:0] temperature;",
            "",
            f"    frozen_intelligence_{safe_model} dut (",
            "        .clk(clk), .rst_n(rst_n), .start(start),",
            "        .busy(busy), .done(done),",
            "        .token_in(token_in), .token_out(token_out),",
            "        .valid_out(valid_out), .new_prompt(new_prompt),",
            "        .temperature(temperature)",
            "    );",
            "",
            f"    always #{period_ns/2} clk = ~clk;",
            "",
            "    initial begin",
            "        clk = 0; rst_n = 0; start = 0; new_prompt = 0;",
            "        token_in = 0; temperature = 128;",
            "        #100;",
            "        rst_n = 1;",
            "        #100;",
            "",
            "        new_prompt = 1;",
            "        token_in = 16'd15043;",
            "        start = 1;",
            "        #20;",
            "        start = 0;",
            "        new_prompt = 0;",
            "",
            "        @(posedge done);",
            "",
            "        new_prompt = 1;",
            "        token_in = 16'd29892;",
            "        start = 1;",
            "        #20;",
            "        start = 0;",
            "        new_prompt = 0;",
            "        @(posedge done);",
            "",
            "        $finish;",
            "    end",
            "",
            "    initial begin",
            "        #1000000;",
            "        $display(\"ERROR: Timeout\");",
            "        $finish;",
            "    end",
            "endmodule",
        ]
        return "\n".join(lines)


class FloorplanEstimator:
    """Estimates physical layout for the chip."""

    CELLS = {
        28: {"mac_um2": 25, "sram_um2": 0.06, "io_um2": 500},
        40: {"mac_um2": 45, "sram_um2": 0.12, "io_um2": 800},
        65: {"mac_um2": 100, "sram_um2": 0.25, "io_um2": 1200},
    }

    def __init__(self, process_nm: int = 28):
        self.process = process_nm
        self.cells = self.CELLS.get(process_nm, self.CELLS[28])

    def estimate_layer_area(self, rows: int, cols: int, bw: int = 4) -> Dict:
        mac_area = rows * cols * self.cells["mac_um2"]
        total = mac_area * 1.3
        return {"rows": rows, "cols": cols, "mac_count": rows * cols,
                "total_area_mm2": round(total * 1e-6, 3)}

    def estimate_kv_cache_area(self, context_length: int, hidden_dim: int,
                                head_dim: int = 64, num_heads: int = 12,
                                bw: int = 16) -> Dict:
        kv_bits = context_length * num_heads * 2 * head_dim * bw
        kv_bytes = kv_bits // 8
        sram_area = kv_bytes * self.cells["sram_um2"]
        return {"kv_bytes": kv_bytes, "sram_area_mm2": round(sram_area * 1e-6, 4)}

    def full_floorplan(self, model_params_b: float, hidden_dim: int = 768,
                       num_layers: int = 12, num_heads: int = 12,
                       context_length: int = 2048, bw: int = 4) -> Dict:
        head_dim = hidden_dim // num_heads
        attn = self.estimate_layer_area(hidden_dim, hidden_dim, bw)
        ffn_up = self.estimate_layer_area(hidden_dim * 4, hidden_dim, bw)
        ffn_down = self.estimate_layer_area(hidden_dim, hidden_dim * 4, bw)
        kv = self.estimate_kv_cache_area(context_length, hidden_dim, head_dim, num_heads, 16)

        compute = (attn["total_area_mm2"] * 4 + ffn_up["total_area_mm2"] +
                   ffn_down["total_area_mm2"]) * num_layers
        sram = kv["sram_area_mm2"] * num_layers
        io = 10.0
        overhead = (compute + sram + io) * 0.2
        total = compute + sram + io + overhead
        side = math.sqrt(total)

        return {
            "process_nm": self.process, "model_params_b": model_params_b,
            "precision": f"INT{bw}", "layers": num_layers,
            "components": {
                "attn_per_layer_mm2": round(attn["total_area_mm2"] * 4, 3),
                "ffn_per_layer_mm2": round(ffn_up["total_area_mm2"] + ffn_down["total_area_mm2"], 3),
                "kv_per_layer_mm2": round(kv["sram_area_mm2"], 4),
            },
            "totals": {
                "compute_mm2": round(compute, 2),
                "sram_mm2": round(sram, 4),
                "die_area_mm2": round(total, 2),
                "die_side_mm": round(side, 2),
            }
        }


def demo():
    print("=== Frozen Intelligence: Verilog & Floorplan Generator ===\n")

    gen = VerilogGenerator(VerilogPrecision.INT4)

    print("--- MAC Unit ---")
    mac = gen.generate_mac_unit()
    print(f"  {len(mac)} chars\n")
    print(mac[:400])
    print()

    print("--- Systolic Array 4x4 ---")
    spec = SystolicArraySpec(rows=4, cols=4, data_width_bits=4)
    arr = gen.generate_systolic_array(spec)
    print(f"  {len(arr)} chars, {spec.mac_count} MACs, {spec.gops:.1f} GOPS\n")

    print("--- Chip Top Module ---")
    layers = {
        "attn_qkv": SystolicArraySpec(12, 64, 4),
        "attn_out": SystolicArraySpec(12, 64, 4),
        "ffn_up": SystolicArraySpec(48, 16, 4),
        "ffn_down": SystolicArraySpec(16, 48, 4),
    }
    top = gen.generate_chip_top("qwen3.5-3b", layers)
    print(f"  {len(top)} chars\n")

    print("--- Testbench ---")
    tb = gen.generate_testbench("qwen3.5-3b")
    print(f"  {len(tb)} chars\n")

    print("=== Floorplan Estimates ===\n")
    fp = FloorplanEstimator(28)
    for name, params, hidden, layers_n, heads in [
        ("Scout", 1.0, 512, 12, 8),
        ("Messenger", 3.0, 768, 24, 12),
        ("Navigator", 7.0, 1024, 32, 16),
        ("Captain", 13.0, 1536, 40, 24),
    ]:
        plan = fp.full_floorplan(params, hidden, layers_n, heads)
        t = plan["totals"]
        print(f"{name} ({params}B, {hidden}d, {layers_n}L):")
        print(f"  Die: {t['die_area_mm2']}mm2 ({t['die_side_mm']}mm x {t['die_side_mm']}mm)")
        print(f"  Compute: {t['compute_mm2']}mm2 | SRAM: {t['sram_mm2']}mm2")
        print()


if __name__ == "__main__":
    demo()
