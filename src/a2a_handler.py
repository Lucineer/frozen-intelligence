#!/usr/bin/env python3
"""A2A protocol handler for frozen intelligence vessels."""
import json, hashlib, time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from enum import Enum


class A2AMessageType(Enum):
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    HEALTH_CHECK = "health_check"
    HEALTH_RESPONSE = "health_response"
    CONFIG_QUERY = "config_query"
    CONFIG_RESPONSE = "config_response"
    THERMAL_ALERT = "thermal_alert"
    SHUTDOWN = "shutdown"
    ACK = "ack"


class A2AQoS(Enum):
    REALTIME = "realtime"       # lowest latency
    RELIABLE = "reliable"       # guaranteed delivery
    BACKGROUND = "background"   # best effort
    BURST = "burst"             # batch processing


@dataclass
class A2AMessage:
    msg_type: A2AMessageType
    source: str
    target: str
    payload: Dict
    qos: A2AQoS = A2AQoS.RELIABLE
    msg_id: str = ""
    timestamp: float = 0.0
    priority: int = 5

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = hashlib.sha256(
                f"{self.source}{self.target}{time.time()}".encode()).hexdigest()[:16]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_wire(self) -> bytes:
        """Serialize for transport (USB/PCIe/SPI)."""
        obj = {
            "id": self.msg_id,
            "type": self.msg_type.value,
            "src": self.source,
            "tgt": self.target,
            "qos": self.qos.value,
            "pri": self.priority,
            "ts": self.timestamp,
            "payload": self.payload,
        }
        return json.dumps(obj).encode()

    @classmethod
    def from_wire(cls, data: bytes) -> "A2AMessage":
        obj = json.loads(data)
        return cls(
            msg_type=A2AMessageType(obj["type"]),
            source=obj["src"],
            target=obj["tgt"],
            payload=obj["payload"],
            qos=A2AQoS(obj.get("qos", "reliable")),
            msg_id=obj.get("id", ""),
            timestamp=obj.get("ts", 0),
            priority=obj.get("pri", 5))


@dataclass
class VesselIdentity:
    vessel_id: str
    vessel_class: str
    model_name: str
    public_key_hash: str
    firmware_version: str
    serial_number: str

    def did(self) -> str:
        """Decentralized Identifier for this vessel."""
        return f"did:frozen:{self.vessel_id}:{self.public_key_hash[:16]}"

    def to_dict(self) -> Dict:
        return {
            "vessel_id": self.vessel_id,
            "vessel_class": self.vessel_class,
            "model": self.model_name,
            "did": self.did(),
            "firmware": self.firmware_version,
            "serial": self.serial_number,
        }


class A2AHandler:
    """Handles A2A protocol for a frozen intelligence vessel."""

    def __init__(self, vessel_id: str, vessel_class: str, model_name: str,
                 inference_fn: Optional[Callable] = None):
        self.identity = VesselIdentity(
            vessel_id=vessel_id,
            vessel_class=vessel_class,
            model_name=model_name,
            public_key_hash=hashlib.sha256(vessel_id.encode()).hexdigest(),
            firmware_version="1.0.0",
            serial_number=f"SN-{vessel_id}")
        self.inference_fn = inference_fn
        self.message_log: List[Dict] = []
        self.request_count = 0
        self.total_tokens = 0

    def handle(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Process incoming A2A message and return response."""
        entry = {
            "id": message.msg_id,
            "type": message.msg_type.value,
            "src": message.source,
            "ts": message.timestamp,
        }

        if message.msg_type == A2AMessageType.HEALTH_CHECK:
            resp = self._health_response(message)
        elif message.msg_type == A2AMessageType.INFERENCE_REQUEST:
            resp = self._inference_response(message)
        elif message.msg_type == A2AMessageType.CONFIG_QUERY:
            resp = self._config_response(message)
        elif message.msg_type == A2AMessageType.SHUTDOWN:
            resp = A2AMessage(
                msg_type=A2AMessageType.ACK,
                source=self.identity.vessel_id,
                target=message.source,
                payload={"shutdown": True, "msg": "Shutting down"},
                qos=message.qos)
        else:
            resp = A2AMessage(
                msg_type=A2AMessageType.ACK,
                source=self.identity.vessel_id,
                target=message.source,
                payload={"error": "unknown_message_type"},
                qos=message.qos)

        entry["response_id"] = resp.msg_id
        self.message_log.append(entry)
        return resp

    def _health_response(self, msg: A2AMessage) -> A2AMessage:
        return A2AMessage(
            msg_type=A2AMessageType.HEALTH_RESPONSE,
            source=self.identity.vessel_id,
            target=msg.source,
            payload={
                "status": "ready",
                "vessel": self.identity.to_dict(),
                "stats": {
                    "requests": self.request_count,
                    "tokens_generated": self.total_tokens,
                    "messages_processed": len(self.message_log),
                },
            },
            qos=msg.qos)

    def _inference_response(self, msg: A2AMessage) -> A2AMessage:
        prompt = msg.payload.get("prompt", "")
        max_tokens = msg.payload.get("max_tokens", 256)
        self.request_count += 1

        if self.inference_fn:
            try:
                output = self.inference_fn(prompt, max_tokens)
                tokens = output.split()
                self.total_tokens += len(tokens)
                status = "success"
            except Exception as e:
                output = f"[error: {str(e)}]"
                tokens = []
                status = "error"
        else:
            output = f"[frozen-intelligence-sim] prompt={prompt[:50]}... model={self.identity.model_name}"
            tokens = output.split()
            self.total_tokens += len(tokens)
            status = "simulated"

        return A2AMessage(
            msg_type=A2AMessageType.INFERENCE_RESPONSE,
            source=self.identity.vessel_id,
            target=msg.source,
            payload={
                "status": status,
                "output": output,
                "tokens": len(tokens),
                "model": self.identity.model_name,
                "vessel_class": self.identity.vessel_class,
                "request_id": msg.msg_id,
            },
            qos=msg.qos)

    def _config_response(self, msg: A2AMessage) -> A2AMessage:
        return A2AMessage(
            msg_type=A2AMessageType.CONFIG_RESPONSE,
            source=self.identity.vessel_id,
            target=msg.source,
            payload={
                "vessel": self.identity.to_dict(),
                "model_config": {
                    "params": {"scout": "1B", "messenger": "3B", "navigator": "7B",
                               "captain": "13B"}.get(self.identity.vessel_class, "unknown"),
                    "quantization": "int4",
                    "process_nm": 28,
                    "max_context_tokens": {"scout": 512, "messenger": 2048,
                                           "navigator": 4096, "captain": 8192}.get(
                        self.identity.vessel_class, 2048),
                },
                "capabilities": {
                    "inference": True,
                    "streaming": True,
                    "kv_cache": True,
                    "fine_tuning": False,
                    "model_swap": False,
                },
            },
            qos=msg.qos)


class A2AFleetBus:
    """Simulates the fleet A2A bus connecting multiple vessels."""

    def __init__(self):
        self.vessels: Dict[str, A2AHandler] = {}

    def register(self, handler: A2AHandler):
        self.vessels[handler.identity.vessel_id] = handler

    def send(self, target: str, msg_type: A2AMessageType,
             payload: Dict, source: str = "fleet-coordinator",
             qos: A2AQoS = A2AQoS.RELIABLE) -> Optional[A2AMessage]:
        if target not in self.vessels:
            return None
        msg = A2AMessage(
            msg_type=msg_type,
            source=source,
            target=target,
            payload=payload,
            qos=qos)
        return self.vessels[target].handle(msg)

    def broadcast_health(self) -> Dict:
        results = {}
        for vid, handler in self.vessels.items():
            resp = self.send(vid, A2AMessageType.HEALTH_CHECK, {})
            if resp:
                results[vid] = resp.payload
        return results

    def broadcast_inference(self, prompt: str, max_tokens: int = 128) -> Dict:
        results = {}
        for vid, handler in self.vessels.items():
            resp = self.send(vid, A2AMessageType.INFERENCE_REQUEST,
                             {"prompt": prompt, "max_tokens": max_tokens})
            if resp:
                results[vid] = resp.payload
        return results


def demo():
    print("=== Frozen Intelligence: A2A Protocol ===\n")

    bus = A2AFleetBus()

    # Register vessels
    for vid, vc, model in [
        ("fi-scout-001", "scout", "qwen3.5-1b-int2"),
        ("fi-messenger-001", "messenger", "qwen3.5-3b-int4"),
        ("fi-navigator-001", "navigator", "qwen3.5-7b-int4"),
    ]:
        bus.register(A2AHandler(vid, vc, model))

    # Health check
    print("--- Fleet Health ---")
    health = bus.broadcast_health()
    for vid, h in health.items():
        print(f"  {vid}: {h['status']} | {h['vessel']['vessel_class']} | {h['vessel']['model']}")
        print(f"    DID: {h['vessel']['did']}")

    # Config query
    print(f"\n--- Config Query: fi-messenger-001 ---")
    cfg = bus.send("fi-messenger-001", A2AMessageType.CONFIG_QUERY, {})
    if cfg:
        print(json.dumps(cfg.payload, indent=2))

    # Inference request
    print(f"\n--- Inference Broadcast ---")
    results = bus.broadcast_inference("What should I check before casting off?", max_tokens=10)
    for vid, r in results.items():
        print(f"  {vid}: [{r['status']}] {r['output'][:60]}...")
        print(f"    tokens: {r['tokens']}, model: {r['model']}")


if __name__ == "__main__":
    demo()
