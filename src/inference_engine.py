#!/usr/bin/env python3
"""Inference engine simulation — emulates mask-locked chip behavior in software."""
import time, json, hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Any
from enum import Enum
from collections import deque


class VesselClass(Enum):
    SCOUT = "scout"         # 1B, 100 tok/s, <1W
    MESSENGER = "messenger"  # 3B, 80 tok/s, 2-3W
    NAVIGATOR = "navigator"  # 7B, 50 tok/s, 4-6W
    CAPTAIN = "captain"     # 13B, 30 tok/s, 8-12W


class ChipState(Enum):
    IDLE = "idle"
    PREFILL = "prefill"
    DECODE = "decode"
    THERMAL_THROTTLE = "thermal_throttle"
    FAULT = "fault"


@dataclass
class ChipConfig:
    vessel_class: VesselClass = VesselClass.MESSENGER
    model_name: str = "qwen3.5-3b-int4"
    interface: str = "usb3"
    process_nm: int = 28
    quantization: str = "int4"
    max_tokens_per_request: int = 2048
    thermal_limit_c: float = 85.0

    @property
    def model_params(self) -> int:
        return {"scout": 1e9, "messenger": 3e9, "navigator": 7e9,
                "captain": 13e9}[self.vessel_class.value]

    @property
    def target_tok_s(self) -> float:
        return {"scout": 100, "messenger": 80, "navigator": 50,
                "captain": 30}[self.vessel_class.value]

    @property
    def target_power_w(self) -> float:
        return {"scout": 0.8, "messenger": 2.5, "navigator": 5.0,
                "captain": 10.0}[self.vessel_class.value]


@dataclass
class TokenResult:
    token: str
    logprob: float
    latency_ms: float
    cumulative_tokens: int
    chip_temp_c: float


class InferenceEngine:
    """Software emulation of a mask-locked inference chip."""

    def __init__(self, config: Optional[ChipConfig] = None,
                 inference_fn: Optional[Callable[[str], str]] = None):
        self.config = config or ChipConfig()
        self.state = ChipState.IDLE
        self.inference_fn = inference_fn  # pluggable LLM backend
        self.kv_cache: Dict = {}
        self.session_id: Optional[str] = None
        self.tokens_generated: int = 0
        self.total_energy_mwh: float = 0.0
        self.thermal_c: float = 25.0
        self.request_count: int = 0
        self._token_buffer: deque = deque()

    def _simulate_latency(self) -> float:
        base = 1000.0 / self.config.target_tok_s
        jitter = base * 0.1
        return base + (hash(str(self.tokens_generated)) % 100 / 100 - 0.5) * jitter

    def _update_thermal(self, active: bool):
        if active:
            self.thermal_c = min(self.thermal_c + 0.3, self.config.thermal_limit_c)
        else:
            self.thermal_c = max(self.thermal_c - 0.1, 25.0)

    def _update_power(self, ms: float):
        watts = self.config.target_power_w if self.state in (ChipState.PREFILL, ChipState.DECODE) else 0.05
        self.total_energy_mwh += watts * ms / 3600

    def begin_session(self, session_id: str):
        self.session_id = session_id
        self.kv_cache.clear()
        self._token_buffer.clear()
        self.state = ChipState.IDLE

    def end_session(self):
        self.session_id = None
        self.kv_cache.clear()
        self._token_buffer.clear()
        self.state = ChipState.IDLE

    def prefill(self, prompt: str) -> int:
        self.state = ChipState.PREFILL
        self._update_thermal(True)
        # Simulate prefill — longer for longer prompts
        tokens_est = len(prompt.split()) * 1.3
        prefill_ms = tokens_est * 0.5  # much faster per token than decode
        self._update_power(prefill_ms)
        self.state = ChipState.DECODE
        self.request_count += 1
        return int(tokens_est)

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop_tokens: Optional[List[str]] = None) -> List[TokenResult]:
        self.begin_session(str(self.request_count))
        prompt_tokens = self.prefill(prompt)
        results = []
        stop_tokens = stop_tokens or ["<|endoftext|>", ".", "\n"]

        for i in range(max_tokens):
            if self.thermal_c >= self.config.thermal_limit_c:
                self.state = ChipState.THERMAL_THROTTLE
                break

            latency_ms = self._simulate_latency()
            self._update_thermal(True)
            self._update_power(latency_ms)
            self.tokens_generated += 1

            # If we have a real inference function, use it (first call only)
            if self.inference_fn and not self._token_buffer:
                full_response = self.inference_fn(prompt)
                self._token_buffer = deque(full_response.split())

            if self._token_buffer:
                token = self._token_buffer.popleft()
            else:
                token = f"tok_{self.tokens_generated}"

            result = TokenResult(
                token=token,
                logprob=round(-0.5 - (hash(token + str(self.tokens_generated)) % 100) / 100, 3),
                latency_ms=round(latency_ms, 2),
                cumulative_tokens=i + 1,
                chip_temp_c=round(self.thermal_c, 1))
            results.append(result)

            if token.strip() in [s.strip() for s in stop_tokens]:
                break

        self._update_thermal(False)
        self.state = ChipState.IDLE
        return results

    def stream_generate(self, prompt: str, max_tokens: int = 256):
        """Generator that yields tokens as they're produced."""
        self.begin_session(str(self.request_count))
        self.prefill(prompt)
        for i in range(max_tokens):
            if self.thermal_c >= self.config.thermal_limit_c:
                break
            latency_ms = self._simulate_latency()
            self._update_thermal(True)
            self._update_power(latency_ms)
            self.tokens_generated += 1
            time.sleep(latency_ms / 1000 * 0.01)  # simulate timing (scaled down)

            if self.inference_fn and not self._token_buffer:
                self._token_buffer = deque(self.inference_fn(prompt).split())

            token = self._token_buffer.popleft() if self._token_buffer else f"tok_{self.tokens_generated}"
            yield TokenResult(
                token=token, logprob=-0.5,
                latency_ms=round(latency_ms, 2),
                cumulative_tokens=i + 1,
                chip_temp_c=round(self.thermal_c, 1))
            if token.strip() in (".", "\n"):
                break
        self._update_thermal(False)
        self.state = ChipState.IDLE

    def health(self) -> Dict:
        return {
            "state": self.state.value,
            "temperature_c": round(self.thermal_c, 1),
            "tokens_generated": self.tokens_generated,
            "total_energy_wh": round(self.total_energy_mwh / 1000, 4),
            "requests": self.request_count,
            "vessel_class": self.config.vessel_class.value,
            "model": self.config.model_name,
            "session": self.session_id,
        }

    def vessel_json(self) -> Dict:
        return {
            "vessel_class": "frozen-intelligence",
            "subclass": self.config.vessel_class.value,
            "model": self.config.model_name,
            "interface": self.config.interface,
            "power_watts": self.config.target_power_w,
            "tokens_per_second": self.config.target_tok_s,
            "quantization": self.config.quantization,
            "process_node": f"{self.config.process_nm}nm",
            "thermal_limit_c": self.config.thermal_limit_c,
        }


class FleetInferenceRouter:
    """Routes inference requests to appropriate frozen intelligence vessels."""

    def __init__(self):
        self.vessels: Dict[str, InferenceEngine] = {}

    def register(self, vessel_id: str, engine: InferenceEngine):
        self.vessels[vessel_id] = engine

    def route(self, prompt: str, preference: Optional[str] = None,
              max_tokens: int = 256) -> Dict:
        if preference and preference in self.vessels:
            engine = self.vessels[preference]
        elif not self.vessels:
            return {"error": "no vessels registered"}
        else:
            # Pick coolest available vessel
            engine = min(self.vessels.values(), key=lambda e: e.thermal_c)
            if engine.state == ChipState.THERMAL_THROTTLE:
                available = [e for e in self.vessels.values()
                             if e.state == ChipState.IDLE]
                if available:
                    engine = available[0]

        vessel_id = [k for k, v in self.vessels.items() if v is engine][0]
        results = engine.generate(prompt, max_tokens)
        return {
            "vessel_id": vessel_id,
            "vessel_class": engine.config.vessel_class.value,
            "model": engine.config.model_name,
            "tokens": [r.token for r in results],
            "stats": {
                "total_tokens": len(results),
                "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 2) if results else 0,
                "final_temp_c": results[-1].chip_temp_c if results else 25.0,
                "energy_wh": round(engine.total_energy_mwh / 1000, 6),
            }
        }

    def fleet_health(self) -> Dict:
        return {vid: v.health() for vid, v in self.vessels.items()}


def demo():
    print("=== Frozen Intelligence: Inference Engine ===\n")

    # Create vessel fleet
    router = FleetInferenceRouter()

    classes = [
        ("fi-scout-001", VesselClass.SCOUT, "qwen3.5-1b-int2"),
        ("fi-messenger-001", VesselClass.MESSENGER, "qwen3.5-3b-int4"),
        ("fi-messenger-002", VesselClass.MESSENGER, "qwen3.5-3b-int4"),
        ("fi-navigator-001", VesselClass.NAVIGATOR, "qwen3.5-7b-int4"),
    ]

    for vid, vc, model in classes:
        cfg = ChipConfig(vessel_class=vc, model_name=model)
        router.register(vid, InferenceEngine(cfg))

    print("Fleet health:")
    for vid, h in router.fleet_health().items():
        print(f"  {vid}: {h['vessel_class']} ({h['model']}) idle@{h['temperature_c']}C")

    # Run inference on messenger
    print(f"\n--- Routing request ---")
    result = router.route("Hello, I am a fishing vessel AI. What should I check?", max_tokens=50)
    print(f"Routed to: {result['vessel_id']} ({result['vessel_class']})")
    print(f"Tokens: {' '.join(result['tokens'][:15])}...")
    print(f"Stats: {result['stats']}")

    # Show thermal after load
    print(f"\nFleet health after request:")
    for vid, h in router.fleet_health().items():
        print(f"  {vid}: {h['state']} {h['temperature_c']}C ({h['tokens_generated']} tokens)")

    # Vessel JSON
    print(f"\n--- vessel.json ---")
    print(json.dumps(router.vessels["fi-messenger-001"].vessel_json(), indent=2))


if __name__ == "__main__":
    demo()
