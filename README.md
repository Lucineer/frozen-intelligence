# frozen-intelligence

Mask-locked inference chip design toolchain. The chip IS the agent.

## Vessel Classes

| Class | Parameters | Power | Tokens/s | Use Case |
|-------|-----------|-------|----------|----------|
| Scout | 1B | <1W | 100 | Edge sensors, microcontrollers |
| Messenger | 3B | 2.5W | 80 | Voice assistants, smart devices |
| Navigator | 7B | 5W | 50 | Field operations, autonomous systems |
| Captain | 13B | 10W | 30 | On-premise reasoning, data center edge |

## Architecture

- **Weight-locked**: Model weights hardwired into metal interconnect
- **Zero boot time**: Powers on and immediately processes tokens
- **Deterministic latency**: Fixed throughput independent of input
- **Mixed precision**: LayerNorm=FP32, Embed=INT8, Attention/FFN=INT4
- **TLMM**: Table-Lookup MatMul (90% LUT reduction vs traditional MAC)
- **Yield-aware**: MoE swarm tiling with defect tolerance and die binning

## Modules (15 Python files, stdlib only)

### Core Toolchain
- `metal_compiler.py` — Weight quantization → binary METL format → die size estimation
- `weight_compiler.py` — Full weight-to-metal pipeline (PyTorch → mixed-precision → binary)
- `inference_engine.py` — Chip simulation, thermal management, fleet routing
- `verilog_generator.py` — MAC units, systolic arrays, chip top module, testbench
- `chip_verifier.py` — Timing/power/Design Rule Check, test pattern generation

### FPGA Prototyping
- `fpga_toolkit.py` — TLMM encoding, COE generation, Hilbert curve weight layout
- `tlmm_engine.py` — Table-Lookup MatMul (arXiv:2510.15926), 4×4 to 64×64 arrays
- `weight_streamer.py` — DDR4 → BRAM streaming controller with pipeline optimization
- `clock_gating.py` — Hierarchical clock gating, DVFS, thermal throttling

### Swarm Architecture
- `swarm_tiler.py` — Yield-aware MoE tiling, defect simulation, die grading (GOLD/SILVER/BRONZE/SCRAP), Monte Carlo yield analysis, Verilog generation, C driver header generation

### Analysis
- `quant_research.py` — Precision sweep, mixed-precision analysis, optimal bits per layer
- `equipment_detector.py` — USB vessel scanning, character sheet generation
- `a2a_handler.py` — A2A protocol, DID identity, fleet bus integration

### SDK & CLI
- `sdk.py` — Host-side USB transport, streaming generation, fleet mode
- `cli.py` — Unified CLI: `compile`, `verify`, `estimate`, `simulate`, `benchmark`

## Quick Start

```bash
cd frozen-intelligence
export PYTHONPATH=src:.

# Run any module demo
python src/tlmm_engine.py
python src/swarm_tiler.py

# CLI usage
python cli.py compile --model qwen3.5-3b --precision int4 --output chip.bin
python cli.py verify --chip messenger --clock 500 --power 3.0
python cli.py estimate --model 3b --process 28nm --yield_pct 85
python cli.py simulate --model messenger --prompt "Hello" --tokens 64
python cli.py benchmark --iterations 10
```

## Swarm Tiling

The swarm architecture tiles multiple specialized models onto a single die:

```
    ┌─────────────────────────┐
    │  EDGE: expendable       │
    │  ┌───────────────────┐  │
    │  │  MID: important   │  │
    │  │  ┌─────────────┐  │  │
    │  │  │  CORE:      │  │  │
    │  │  │  Router     │  │  │
    │  │  │  Safety     │  │  │
    │  │  │  Primary    │  │  │
    │  │  └─────────────┘  │  │
    │  └───────────────────┘  │
    └─────────────────────────┘
```

Defective tiles are disabled. MoE router skips dead tiles. Each die is binned:
- **GOLD**: Full swarm → flagship price
- **SILVER**: Core + most experts → standard price
- **BRONZE**: Core only → budget SKU
- **SCRAP**: Non-functional

## Zero Dependencies

All modules use Python stdlib only. No PyTorch, no TensorFlow, no external packages.
Runs on Jetson ARM64, Raspberry Pi, or any Python 3.11+ environment.

## Research

See `docs/research/` for 5 DeepSeek RA rounds (~80K chars):
- Weight-to-metal compilation
- Manufacturing process
- Software SDK design
- Competitive analysis
- Technical risks

## License

MIT — Lucineer (DiGennaro et al.)
