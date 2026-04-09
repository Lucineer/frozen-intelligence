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
- **NoC mesh**: XY routing with virtual channels and wormhole switching
- **DDR4/LPDDR4**: Bank-interleaved memory controller with refresh management

## Modules (31 source + CLI + tests, stdlib only)

### Core Toolchain
1. `metal_compiler.py` — Weight quantization → binary METL format → die size
2. `weight_compiler.py` — Full weight-to-metal pipeline (PyTorch → mixed-precision → binary)
3. `inference_engine.py` — Chip simulation, thermal management, fleet routing
4. `verilog_generator.py` — MAC units, systolic arrays, chip top module, testbench
5. `chip_verifier.py` — Timing/power/DRC, test pattern generation

### Physical Design
6. `gdsii_generator.py` — GDSII stream format: weight tiles, pad rings, die assembly
7. `drc_checker.py` — Design Rule Checker: width, spacing, enclosure, density (28nm LP)
8. `floorplanner.py` — Chip floorplan: weight banks, I/O pads, power grid, clock tree, thermal
9. `fpga_toolkit.py` — TLMM encoding, COE generation, Hilbert curve weight layout

### FPGA Prototyping
10. `tlmm_engine.py` — Table-Lookup MatMul (arXiv:2510.15926), 4×4 to 64×64 arrays
11. `weight_streamer.py` — DDR4 → BRAM streaming controller with pipeline
12. `clock_gating.py` — Hierarchical clock gating, DVFS, thermal throttling
13. `pcie_interface.py` — PCIe config space, BAR mapping, MMIO, DMA simulation

### Swarm Architecture
14. `swarm_tiler.py` — Yield-aware MoE tiling, die grading (GOLD/SILVER/BRONZE/SCRAP), Monte Carlo

### Digital Design
15. `netlist_gen.py` — Structural Verilog netlist: MAC, multiplier, weight bank, top-level
16. `testbench_gen.py` — Verilog testbench: stimulus, reset, scoreboard, response checking
17. `layer_simulator.py` — NN layer simulation with mixed precision and per-vessel throughput

### Analysis
18. `quant_research.py` — Precision sweep, mixed-precision analysis, optimal bits per layer
19. `timing_analyzer.py` — STA: MAC critical path, systolic arrays, PVT corner analysis
20. `power_estimator.py` — Dynamic + leakage power, per-vessel, process node comparison
21. `signal_integrity.py` — Eye diagrams, crosstalk, termination, transmission line modeling

### Synthesis
20. `synth_estimator.py` — Cell library area/power estimation, multiplier, systolic, full-chip

### On-Chip Infrastructure
21. `memory_controller.py` — DDR4/LPDDR4: command scheduling, bank interleaving, refresh
22. `noc_router.py` — Mesh NoC: XY routing, virtual channels, wormhole switching

### Integration & Security
23. `equipment_detector.py` — USB vessel scanning, character sheet generation
24. `a2a_handler.py` — A2A protocol, DID identity, fleet bus integration
25. `hardware_security.py` — TRNG, AES-256 CTR, secure boot chain, PUF chip authentication

### Verification
24. `formal_checker.py` — Bounded model checking: safety, liveness, invariants, counterexamples
25. `rtl_simulator.py` — Event-driven RTL sim: AND/OR/XOR/MUX/FA/DFF, trace, multi-cycle

### Cost & Debug
26. `cost_model.py` — NRE, unit cost, yield, packaging, volume/process comparison
27. `jtag_debug.py` — JTAG TAP controller, DAP debug access, boundary scan

### SDK & CLI
28. `sdk.py` — Host-side USB transport, streaming generation, fleet mode
29. `cli.py` — Unified CLI: `compile`, `verify`, `estimate`, `simulate`, `benchmark`

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
