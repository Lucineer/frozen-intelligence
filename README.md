# frozen-intelligence — Mask-Locked Inference Chip Toolchain

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/frozen-intelligence.git
cd frozen-intelligence

# Run tests
python3 tests/test_all.py

# Use CLI
python3 cli.py --help
python3 cli.py simulate --prompt "Hello" --tokens 16
python3 cli.py estimate --model 3b --process 28nm --yield_pct 85
```

## Docker

```bash
# Build
docker build -t frozen-intelligence .

# Run
docker run -it frozen-intelligence --help
docker run -it frozen-intelligence simulate --prompt "Hello"
```

## Architecture

The chip IS the agent. Weight-locked inference with:

- **TLMM**: Table-Lookup MatMul (90% LUT reduction vs traditional MAC)
- **Yield-aware swarm tiling**: MoE routing with defect tolerance
- **Zero boot time**: Powers on and immediately processes tokens
- **Mixed precision**: LayerNorm=FP32, Embed=INT8, Attention/FFN=INT4

## Vessel Classes

| Class | Parameters | Power | Tokens/s | Use Case |
|-------|-----------|-------|----------|----------|
| Scout | 1B | <1W | 100 | Edge sensors, microcontrollers |
| Messenger | 3B | 2.5W | 80 | Voice assistants, smart devices |
| Navigator | 7B | 5W | 50 | Field operations, autonomous systems |
| Captain | 13B | 10W | 30 | On-premise reasoning, data center edge |

## Modules

15 Python modules, stdlib only:

1. `metal_compiler.py` — Weight quantization → binary METL format
2. `weight_compiler.py` — Full weight-to-metal pipeline
3. `inference_engine.py` — Chip simulation, thermal management
4. `verilog_generator.py` — MAC units, systolic arrays, testbench
5. `chip_verifier.py` — Timing/power/DRC, test patterns
6. `fpga_toolkit.py` — TLMM encoding, COE generation, Hilbert layout
7. `tlmm_engine.py` — Table-Lookup MatMul (arXiv:2510.15926)
8. `weight_streamer.py` — DDR4 → BRAM streaming controller
9. `clock_gating.py` — Hierarchical clock gating, DVFS, thermal throttling
10. `swarm_tiler.py` — Yield-aware MoE tiling, die grading, Monte Carlo yield
11. `quant_research.py` — Precision sweep, mixed-precision analysis
12. `equipment_detector.py` — USB vessel scanning, character sheet
13. `a2a_handler.py` — A2A protocol, DID identity, fleet bus
14. `sdk.py` — Host-side USB transport, streaming generation, fleet mode
15. `cli.py` — Unified CLI: compile/verify/estimate/simulate/benchmark

## Swarm Tiling

Yield-aware Mixture-of-Experts:

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

## License

MIT — Lucineer (DiGennaro et al.)
