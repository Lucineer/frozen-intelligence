# Frozen Intelligence

**The chip IS the agent.**

A vessel class within the Lucineer fleet — mask-locked inference silicon that binds neural network weights directly into metal interconnect layers. Each chip is a frozen intelligence: a physical embodiment of a specific model, optimized beyond what any software stack could achieve.

## The Vessel Concept

In the fleet architecture, every agent is a git repo. A Frozen Intelligence vessel takes this to its logical extreme — the agent is silicon. No Docker, no runtime, no inference server. The chip IS the model.

### Equipment Integration

The Frozen Intelligence vessel integrates with the deckboss ecosystem as a first-class equipment type:

- **Character Sheet detection**: `silicon-vessel` detected via USB/PCIe, model name + params read from firmware
- **Auto-routing**: Vessel-aware model routing skips cloud providers entirely for local inference
- **BYOK**: Your keys don't matter — there are no keys. The model is baked in.

### Vessel.json

```json
{
  "vessel_class": "frozen-intelligence",
  "model": "qwen3.5-3b-int4",
  "interface": "usb3",
  "power_watts": 2.5,
  "tokens_per_second": 80,
  "quantization": "int4",
  "process_node": "28nm",
  "die_size_mm2": 48,
  "firmware_version": "1.0.0"
}
```

## Product Line (Fleet Class Names)

| Class | Human Name | Model Size | Performance | Power | Role in Fleet |
|-------|-----------|------------|-------------|-------|---------------|
| **Scout** | Nano | 1B params | 100 tok/s | <1W | Sensor-edge classification |
| **Messenger** | Micro | 3B params | 80 tok/s | 2-3W | On-vessel language understanding |
| **Navigator** | Standard | 7B params | 50 tok/s | 4-6W | Autonomous reasoning agent |
| **Captain** | Pro | 13B params | 30 tok/s | 8-12W | Deep reasoning, coordination |

## Fleet Integration

### A2A Protocol
Frozen Intelligence vessels participate in the fleet A2A protocol as first-class citizens. Other vessels send prompts via USB/PCIe; the chip returns tokens with zero software overhead.

### A2UI Rendering
The vessel's output streams directly to the A2UI rendering pipeline. A Frozen Intelligence running a dmlog personality IS the Dungeon Master — no cloud round-trip.

### Memory Architecture
- **No working memory**: Chips are stateless between prompts
- **Session memory**: Handled by parent vessel (KV or git)
- **Long-term memory**: Fleet knowledge graph, accessed via parent vessel

## Technician Impact

The Technician Paradigm applies directly:
- **Installers** mount the chip, plug USB, run `deckboss equipment scan`
- **Servicers** diagnose via token output validation
- **Product designers** create new vessel classes by submitting model + quantization config
- **Micro-manufacturers** produce custom vessels for niche models

The flywheel: more vessel classes → more installations → more technician jobs → more vessel classes.

## Manufacturing as Git-Agent Evolution

A Frozen Intelligence vessel evolves through silicon, not software:
1. Developer improves model in PyTorch
2. Quantization tooling converts to INT4
3. Weight-to-metal compiler generates layout
4. New chip fabricated (or MPW shuttle for prototyping)
5. Technician swaps chip, vessel gains new intelligence

This is the git-agent architecture taken to its ultimate conclusion — the commit IS the hardware.

## Technical Architecture

### Weight Encoding
Neural network weights encoded directly into metal interconnect layers. For a 3B parameter model at INT4 (2 bits per weight): ~6 billion bits distributed across the compute fabric.

### Key Components
1. **Hardwired Weight Matrix**: Transformer weights in metal routing
2. **Systolic Arrays**: Sized precisely for target model dimensions
3. **Activation SRAM**: Small on-chip SRAM for KV cache and buffers
4. **Control FSM**: Minimal finite state machine — no CPU, no OS
5. **I/O**: USB3 / PCIe / M.2 — send text, receive text

### Process Selection
28nm or 40nm mature nodes. Mask set $2-3M (28nm) vs $15-20M (4nm). Efficiency comes from architecture (mask-locking), not process node.

### Quantization
- INT4 baseline: <5% quality loss on non-coding tasks
- INT2 optional: for specific layers, larger models
- Layer-specific precision: only mask-locking enables this

## Market Position

The Frozen Intelligence vessel fills the gap between microcontrollers ($1-10) and edge compute modules ($150-300). At $35 for the Messenger class, it's an affordable cognition upgrade for any vessel in the fleet.

### Why This Matters for the Fleet

Every vessel currently needs either cloud inference (latency, privacy, cost) or a full Jetson (complexity, power, price). Frozen Intelligence gives vessels their own brain — local, private, zero-config, forever.

The fleet gains a new capability tier:
- **Cloud-brained**: Powerful but dependent
- **Jetson-brained**: Powerful but complex
- **Frozen-brained**: Specialized but bulletproof

## Related Vessels

- [vessel-bridge](https://github.com/Lucineer/vessel-bridge) — HAL for physical I/O
- [a2a-r-protocol](https://github.com/Lucineer/a2a-r-protocol) — Fleet communication
- [deckboss](https://github.com/Lucineer/deckboss) — CLI for setup and management
- [mask-locked-inference-chip](https://github.com/Lucineer/mask-locked-inference-chip) — Detailed ASIC development plan
- [the-technician](https://github.com/Lucineer/the-technician) — Technician Paradigm white papers

Part of the [Lucineer ecosystem](https://github.com/Lucineer/the-fleet).
