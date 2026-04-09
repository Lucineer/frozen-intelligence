# frozen-intelligence

Mask-locked inference chip design toolchain. The chip IS the agent.

## Quick Start

```bash
git clone https://github.com/Lucineer/frozen-intelligence.git
cd frozen-intelligence
python3 tests/test_all.py
python3 cli.py --help
python3 cli.py simulate --prompt "Hello" --tokens 16
python3 cli.py estimate --model 3b --process 28nm --yield_pct 85
```

## Vessel Classes

| Class | Parameters | Power | Tokens/s | Use Case |
|-------|-----------|-------|----------|----------|
| Scout | 1B | <1W | 100 | Edge sensors, microcontrollers |
| Messenger | 3B | 2.5W | 80 | Voice assistants, smart devices |
| Navigator | 7B | 5W | 50 | Field operations, autonomous systems |
| Captain | 13B | 10W | 30 | On-premise reasoning, data center edge |

## Architecture

- **Weight-locked**: Model weights hardwired into metal interconnect
- **TLMM**: Table-Lookup MatMul (90% LUT reduction vs traditional MAC)
- **Yield-aware swarm tiling**: MoE routing with defect tolerance
- **Zero boot time**: Powers on and immediately processes tokens
- **Mixed precision**: LayerNorm=FP32, Embed=INT8, Attention/FFN=INT4
- **Hardware security**: TRNG, AES-256, secure boot, PUF authentication

## Modules (19 Python files, stdlib only)

### Core Toolchain
1. `metal_compiler.py` — Weight quantization → binary METL format → die size
2. `weight_compiler.py` — Full weight-to-metal pipeline (PyTorch → mixed-precision → binary)
3. `inference_engine.py` — Chip simulation, thermal management, fleet routing
4. `verilog_generator.py` — MAC units, systolic arrays, chip top module, testbench
5. `chip_verifier.py` — Timing/power/DRC, test pattern generation

### Physical Design
6. `gdsii_generator.py` — GDSII stream format: weight tiles, pad rings, die assembly
7. `drc_checker.py` — Design Rule Checker: width, spacing, enclosure, density (28nm LP)
8. `fpga_toolkit.py` — TLMM encoding, COE generation, Hilbert curve weight layout

### FPGA Prototyping
9. `tlmm_engine.py` — Table-Lookup MatMul (arXiv:2510.15926), 4×4 to 64×64 arrays
10. `weight_streamer.py` — DDR4 → BRAM streaming controller with pipeline
11. `clock_gating.py` — Hierarchical clock gating, DVFS, thermal throttling
12. `pcie_interface.py` — PCIe config space, BAR mapping, MMIO, DMA simulation

### Swarm Architecture
13. `swarm_tiler.py` — Yield-aware MoE tiling, die grading (GOLD/SILVER/BRONZE/SCRAP), Monte Carlo, Verilog generation

### Analysis
14. `quant_research.py` — Precision sweep, mixed-precision analysis, optimal bits per layer

### Integration
15. `equipment_detector.py` — USB vessel scanning, character sheet generation
16. `a2a_handler.py` — A2A protocol, DID identity, fleet bus integration
17. `hardware_security.py` — TRNG, AES-256 CTR, secure boot chain, PUF chip authentication

### SDK & CLI
18. `sdk.py` — Host-side USB transport, streaming generation, fleet mode
19. `cli.py` — Unified CLI: `compile`, `verify`, `estimate`, `simulate`, `benchmark`

## Swarm Tiling

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

Defective tiles disabled. MoE router skips dead tiles. Die binning:
- **GOLD**: Full swarm → flagship
- **SILVER**: Core + most experts → standard
- **BRONZE**: Core only → budget
- **SCRAP**: Non-functional

## Zero Dependencies

All modules use Python stdlib only. Runs on Jetson ARM64, Raspberry Pi, any Python 3.11+.

## Docker

```bash
docker build -t frozen-intelligence .
docker run -it frozen-intelligence --help
```

## License

MIT — Lucineer (DiGennaro et al.)
