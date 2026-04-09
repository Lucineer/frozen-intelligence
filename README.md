# Frozen Intelligence

**The chip IS the agent.**

Mask-locked inference silicon that binds neural network weights directly into metal interconnect layers. Each chip is a frozen intelligence: a physical embodiment of a specific model, optimized beyond what any software stack could achieve.

## Source Code

### `src/metal_compiler.py` — Weight-to-Metal Compiler
Converts trained model weights to quantized metal encoding. The pipeline that would drive mask fabrication.

- **WeightQuantizer**: FP32/FP16/INT8/INT4/INT2/Binary quantization with per-layer scale computation
- **MetalEncoder**: Binary chip format with header, layer table, and weight data
- **ChipEstimator**: Die area, power, and throughput estimation for 28/40/65nm processes

```python
from metal_compiler import WeightQuantizer, MetalEncoder, ChipEstimator, Precision

q = WeightQuantizer(Precision.INT4)
weights = q.quantize(model_weights, layer_name="attention_qkv", shape=(12, 256, 768))

enc = MetalEncoder()
chip_binary = enc.encode_chip({"attention_qkv": weights}, metadata={"model": "qwen3.5-3b"})

est = ChipEstimator(process_nm=28)
report = est.full_report(model_params_b=3.0, bits=4)
# Die: 595mm², Power: 0.11W, Throughput: 0.1 tok/s (estimate)
```

### `src/inference_engine.py` — Inference Engine Simulation
Software emulation of a mask-locked chip's behavior.

- **InferenceEngine**: Token-by-token generation with thermal simulation, KV cache, power tracking
- **FleetInferenceRouter**: Multi-vessel routing with load balancing and thermal awareness
- **VesselClass**: Scout/Messenger/Navigator/Captain with realistic specs

```python
from inference_engine import InferenceEngine, ChipConfig, VesselClass, FleetInferenceRouter

router = FleetInferenceRouter()
router.register("fi-msg-001", InferenceEngine(ChipConfig(VesselClass.MESSENGER)))
result = router.route("Cast off heading north", max_tokens=50)
```

### `src/equipment_detector.py` — Equipment Detection
Integrates with deckboss CLI for auto-discovery of connected vessels.

- **VesselScanner**: USB/PCIe scanning with mock mode for testing
- **DeckbossIntegration**: Character sheet generation from detected hardware
- **DetectedVessel**: Full vessel metadata including serial, firmware, die specs

```python
from equipment_detector import DeckbossIntegration
integration = DeckbossIntegration()
result = integration.scan_equipment(mock=True)
# Returns vessel configs for character sheet patch
```

### `src/a2a_handler.py` — A2A Protocol Handler
Fleet communication protocol for frozen intelligence vessels.

- **A2AHandler**: Inference request/response, health checks, config queries
- **A2AFleetBus**: Multi-vessel message routing with broadcast
- **VesselIdentity**: DID-based cryptographic identity per vessel

```python
from a2a_handler import A2AHandler, A2AFleetBus, A2AMessageType, A2AQoS

bus = A2AFleetBus()
bus.register(A2AHandler("fi-msg-001", "messenger", "qwen3.5-3b-int4"))
resp = bus.send("fi-msg-001", A2AMessageType.INFERENCE_REQUEST,
                {"prompt": "Hello", "max_tokens": 50}, qos=A2AQoS.REALTIME)
```

## Vessel Classes

| Class | Human Name | Model | Performance | Power | Role |
|-------|-----------|-------|-------------|-------|------|
| Scout | Nano | 1B | 100 tok/s | <1W | Sensor-edge |
| Messenger | Micro | 3B | 80 tok/s | 2-3W | On-vessel reasoning |
| Navigator | Standard | 7B | 50 tok/s | 4-6W | Autonomous agent |
| Captain | Pro | 13B | 30 tok/s | 8-12W | Deep reasoning |

## The Vessel Concept

In the fleet architecture, every agent is a git repo. A Frozen Intelligence vessel takes this to its logical extreme — the agent is silicon. No Docker, no runtime, no inference server.

### vessel.json

```json
{
  "vessel_class": "frozen-intelligence",
  "subclass": "messenger",
  "model": "qwen3.5-3b-int4",
  "interface": "usb3",
  "power_watts": 2.5,
  "tokens_per_second": 80,
  "quantization": "int4",
  "process_node": "28nm"
}
```

## Technician Integration

- **Installers** mount chip, plug USB, run `deckboss equipment scan`
- **Servicers** diagnose via token output validation
- **Product designers** create vessel classes by submitting model + quantization config
- **Micro-manufacturers** produce custom vessels for niche models

## Related

- [vessel-bridge](https://github.com/Lucineer/vessel-bridge) — HAL for physical I/O
- [a2a-r-protocol](https://github.com/Lucineer/a2a-r-protocol) — Fleet communication
- [deckboss](https://github.com/Lucineer/deckboss) — CLI for setup and management
- [mask-locked-inference-chip](https://github.com/Lucineer/mask-locked-inference-chip) — ASIC development plan
- [the-technician](https://github.com/Lucineer/the-technician) — Technician Paradigm

Part of the [Lucineer ecosystem](https://github.com/Lucineer/the-fleet).
