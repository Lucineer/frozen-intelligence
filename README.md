# frozen-intelligence

> The chip IS the agent. No CPU, no OS, no software stack — neural network weights are hardwired into silicon interconnect, creating a self-contained inference vessel.

## Vessel Classes

| Class | Params | Clock | Power | Tokens/s | Process |
|-------|--------|-------|-------|----------|---------|
| Scout | 1B | 500MHz | 1W | 100 | 28nm |
| Messenger | 3B | 500MHz | 3W | 80 | 28nm |
| Navigator | 7B | 500MHz | 6W | 50 | 28nm |
| Captain | 13B | 500MHz | 12W | 30 | 28nm |

## Architecture

- **Weight-to-metal**: Quantized weights encoded directly as metal interconnect geometry
- **Zero runtime**: No CPU, no memory fetch, no software — the silicon IS the model
- **Mixed precision**: LayerNorm=FP32, Embeddings=INT8, Attention+FFN=INT4
- **Systolic arrays**: Dataflow architecture with pipelined MAC units
- **Thermal management**: Hardware-only DVFS with temperature sensors

## Python Modules

- `metal_compiler.py (weight quantization + binary chip encoding)`
- `inference_engine.py (chip simulation, thermal throttling, fleet routing)`
- `equipment_detector.py (USB vessel scanning + character sheet)`
- `a2a_handler.py (A2A protocol, DID identity, fleet bus)`
- `verilog_generator.py (MAC unit, systolic arrays, chip top, testbench, floorplan)`
- `chip_verifier.py (timing analysis, power checks, design rules, test patterns)`
- `quant_research.py (precision sweep, mixed-precision analysis, optimal bits)`
- `weight_compiler.py (full weight-to-metal compilation pipeline)`

## Research

- Weight-to-metal compilation pipeline
- Manufacturing playbook (MPW → production)
- Software SDK design (USB driver, streaming protocol)
- Competitive analysis vs NVIDIA/Hailo/Groq
- Technical risk assessment

## Quick Start

```bash
# Generate Verilog for a chip
python3 src/verilog_generator.py

# Compile weights to mask format
python3 src/weight_compiler.py

# Verify chip meets timing/power
python3 src/chip_verifier.py

# Estimate quantization quality
python3 src/quant_research.py

# Plan tapeout costs
python3 src/tapeout_planner.py  # in mask-locked-inference-chip repo
```

## License

MIT — Lucineer / SuperInstance / DiGennaro et al.
