#!/usr/bin/env python3
"""Frozen Intelligence SDK — host-side Python API for mask-locked chips via USB."""
import struct, time, math, json, os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
from collections import deque


class ChipState(Enum):
    IDLE = 0
    BUSY = 1
    STREAMING = 2
    ERROR = 3
    OVERHEAT = 4
    THERMAL_THROTTLE = 5


class VesselClass(Enum):
    SCOUT = "scout"        # 1B params
    MESSENGER = "messenger" # 3B params
    NAVIGATOR = "navigator" # 7B params
    CAPTAIN = "captain"     # 13B params

    @classmethod
    def from_id(cls, chip_id: int) -> Optional["VesselClass"]:
        mapping = {0x10: cls.SCOUT, 0x20: cls.MESSENGER,
                   0x30: cls.NAVIGATOR, 0x40: cls.CAPTAIN}
        return mapping.get(chip_id)


@dataclass
class ChipInfo:
    vessel_class: VesselClass
    model_name: str
    firmware_version: str
    serial_number: str
    max_context: int
    max_tokens: int
    clock_mhz: int
    process_nm: int
    precision_bits: int


@dataclass
class TokenStream:
    tokens: List[int]
    text: str
    done: bool
    prompt_tokens: int = 0
    generated_tokens: int = 0
    elapsed_ms: float = 0

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_ms > 0:
            return self.generated_tokens / (self.elapsed_ms / 1000)
        return 0


@dataclass
class HealthReport:
    temperature_c: float
    power_w: float
    voltage_v: float
    clock_mhz: float
    state: ChipState
    total_tokens_generated: int
    total_uptime_hours: float
    thermal_throttle_count: int
    error_count: int

    def to_dict(self) -> Dict:
        return {"temperature_c": round(self.temperature_c, 1),
                "power_w": round(self.power_w, 2),
                "voltage_v": round(self.voltage_v, 3),
                "clock_mhz": round(self.clock_mhz, 0),
                "state": self.state.name,
                "total_tokens": self.total_tokens_generated,
                "uptime_hours": round(self.total_uptime_hours, 1),
                "throttle_count": self.thermal_throttle_count,
                "errors": self.error_count}


class SimulatedUSBTransport:
    """Simulates USB3 bulk transfer for development without hardware."""

    def __init__(self, chip_info: ChipInfo, latency_ms: float = 5.0):
        self.chip_info = chip_info
        self.latency = latency_ms / 1000
        self._state = ChipState.IDLE
        self._temp = 45.0
        self._power = 0.5
        self._voltage = 3.3
        self._clock = float(chip_info.clock_mhz)
        self._total_tokens = 0
        self._throttle_count = 0
        self._error_count = 0
        self._start_time = time.time()
        self._last_activity = time.time()

    def write(self, data: bytes) -> int:
        time.sleep(self.latency)
        self._last_activity = time.time()
        return len(data)

    def read(self, size: int, timeout_ms: int = 5000) -> bytes:
        time.sleep(self.latency)
        self._last_activity = time.time()
        return b"\x00" * size

    def write_read(self, data: bytes, read_size: int, timeout_ms: int = 5000) -> bytes:
        self.write(data)
        resp = bytearray(read_size)
        if read_size >= 1 and data and data[0] in (CMD_PING, CMD_IDENTIFY, CMD_HEALTH, CMD_RESET):
            resp[0] = RESP_OK
        if read_size >= 2 and data and data[0] == CMD_PING:
            resp[1] = 0x01
        if data and data[0] == CMD_HEALTH and read_size >= 24:
            struct.pack_into("<ffHH", resp, 1,
                            self._temp, self._power, int(self._voltage * 1000), int(self._clock))
            resp[12] = ChipState.IDLE.value
            struct.pack_into("<I", resp, 13, self._total_tokens)
            uptime = int((time.time() - self._start_time) * 3600)
            struct.pack_into("<I", resp, 17, uptime)
            struct.pack_into("<H", resp, 21, self._throttle_count)
            struct.pack_into("<H", resp, 23, self._error_count)
        return bytes(resp)

    def close(self):
        pass


# USB protocol constants
CMD_PING = 0x01
CMD_IDENTIFY = 0x02
CMD_GENERATE = 0x10
CMD_STREAM_START = 0x11
CMD_STREAM_DATA = 0x12
CMD_STREAM_STOP = 0x13
CMD_CONFIGURE = 0x20
CMD_SET_TEMPERATURE = 0x21
CMD_SET_MAX_TOKENS = 0x22
CMD_HEALTH = 0x30
CMD_RESET = 0x40
CMD_FIRMWARE_UPDATE = 0x50

RESP_OK = 0x80
RESP_ERROR = 0xFF
RESP_TOKEN = 0x81
RESP_STREAM_END = 0x82
RESP_STREAM_ERROR = 0x83


class FrozenIntelligenceSDK:
    """Main SDK class for interacting with mask-locked inference chips."""

    # Default tokenizer (simple BPE-like mapping for demo)
    DEFAULT_TOKEN_MAP = {
        2: "",       # EOS
        15043: "Hello",
        29892: "The",
        29915: " is",
        29871: " the",
        279: ".",
        11: ",",
        319: " and",
        323: " or",
        13: "\n",
        12: " ",
    }

    def __init__(self, transport=None, port: str = "/dev/ttyUSB0"):
        self.port = port
        if transport:
            self._transport = transport
        else:
            info = ChipInfo(vessel_class=VesselClass.SCOUT, model_name="qwen3.5-1b",
                          firmware_version="0.1.0", serial_number="FI-001",
                          max_context=2048, max_tokens=512,
                          clock_mhz=500, process_nm=28, precision_bits=4)
            self._transport = SimulatedUSBTransport(info)
        self._token_map = dict(self.DEFAULT_TOKEN_MAP)
        self._on_token_callbacks: List[Callable[[int, str], None]] = []
        self._on_error_callbacks: List[Callable[[str], None]] = []

    def connect(self) -> bool:
        """Establish USB connection to chip."""
        try:
            ping = self._transport.write_read(
                struct.pack("BB", CMD_PING, 0x00), 2)
            return ping[0] == RESP_OK
        except Exception:
            return False

    def identify(self) -> Optional[ChipInfo]:
        """Read chip identification."""
        try:
            resp = self._transport.write_read(
                struct.pack("BB", CMD_IDENTIFY, 0x00), 64)
            if resp[0] != RESP_OK:
                return None
            # Parse response
            cls = VesselClass.from_id(resp[1])
            name_len = resp[2]
            model_name = resp[3:3 + name_len].decode("ascii", errors="replace")
            fw = f"{resp[3+name_len]}.{resp[4+name_len]}.{resp[5+name_len]}"
            serial = resp[6+name_len:22+name_len].decode("ascii", errors="replace")
            max_ctx, max_tok = struct.unpack("<HH", resp[22+name_len:26+name_len])
            clock, proc, prec = resp[26+name_len], resp[27+name_len], resp[28+name_len]
            return ChipInfo(vessel_class=cls or VesselClass.SCOUT, model_name=model_name,
                          firmware_version=fw, serial_number=serial,
                          max_context=max_ctx, max_tokens=max_tok,
                          clock_mhz=clock, process_nm=proc, precision_bits=prec)
        except Exception:
            return None

    def on_token(self, callback: Callable[[int, str], None]):
        """Register callback for each generated token."""
        self._on_token_callbacks.append(callback)

    def on_error(self, callback: Callable[[str], None]):
        """Register callback for errors."""
        self._on_error_callbacks.append(callback)

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7, stream: bool = True) -> TokenStream:
        """Generate text from prompt."""
        tokens = self._encode(prompt)
        if not tokens:
            tokens = [29892]  # "The"

        start_time = time.time()
        result_tokens = []
        result_text = ""

        if stream:
            # Send stream start
            cmd = struct.pack("<BBHH", CMD_STREAM_START, len(tokens),
                            max_tokens, int(temperature * 255))
            for t in tokens:
                cmd += struct.pack("<H", t)
            self._transport.write(cmd)

            # Read streaming tokens
            for _ in range(max_tokens):
                resp = self._transport.write_read(
                    struct.pack("BB", CMD_STREAM_DATA, 0x00), 4, timeout_ms=10000)
                if resp[0] == RESP_STREAM_END:
                    break
                if resp[0] == RESP_TOKEN:
                    token = struct.unpack("<H", resp[1:3])[0]
                    result_tokens.append(token)
                    text = self._decode_token(token)
                    result_text += text
                    for cb in self._on_token_callbacks:
                        cb(token, text)
                elif resp[0] == RESP_STREAM_ERROR:
                    for cb in self._on_error_callbacks:
                        cb("Stream error from chip")
                    break
        else:
            # Non-streaming generation
            cmd = struct.pack("<BBHH", CMD_GENERATE, len(tokens),
                            max_tokens, int(temperature * 255))
            for t in tokens:
                cmd += struct.pack("<H", t)
            resp = self._transport.write_read(cmd, max_tokens * 2 + 4, timeout_ms=30000)
            if resp[0] == RESP_OK:
                count = struct.unpack("<H", resp[1:3])[0]
                for i in range(count):
                    t = struct.unpack("<H", resp[3 + i * 2:5 + i * 2])[0]
                    result_tokens.append(t)
                    result_text += self._decode_token(t)

        elapsed = (time.time() - start_time) * 1000
        return TokenStream(tokens=result_tokens, text=result_text, done=True,
                          prompt_tokens=len(tokens), generated_tokens=len(result_tokens),
                          elapsed_ms=elapsed)

    def health(self) -> HealthReport:
        """Read chip health status."""
        resp = self._transport.write_read(
            struct.pack("BB", CMD_HEALTH, 0x00), 32)
        if resp[0] != RESP_OK or len(resp) < 24:
            return self._simulated_health()
        temp, power, voltage, clock = struct.unpack("<ffHH", resp[1:13])
        state_byte = resp[13]
        total_tok = struct.unpack("<I", resp[13:17])[0]
        uptime = struct.unpack("<I", resp[17:21])[0] / 3600
        throttle = struct.unpack("<H", resp[21:23])[0]
        errors = struct.unpack("<H", resp[23:25])[0]
        return HealthReport(temperature_c=temp, power_w=power, voltage_v=voltage,
                          clock_mhz=clock, state=ChipState(state_byte),
                          total_tokens_generated=total_tok, total_uptime_hours=uptime,
                          thermal_throttle_count=throttle, error_count=errors)

    def _simulated_health(self) -> HealthReport:
        t = self._transport
        if isinstance(t, SimulatedUSBTransport):
            elapsed = time.time() - t._start_time
            t._temp += 0.1 * (50 - t._temp)  # drift toward 50C
            return HealthReport(temperature_c=t._temp, power_w=t._power + 1.0,
                              voltage_v=t._voltage, clock_mhz=t._clock,
                              state=ChipState.IDLE, total_tokens_generated=t._total_tokens,
                              total_uptime_hours=elapsed / 3600,
                              thermal_throttle_count=t._throttle_count,
                              error_count=t._error_count)
        return HealthReport(temperature_c=0, power_w=0, voltage_v=0, clock_mhz=0,
                          state=ChipState.IDLE, total_tokens_generated=0,
                          total_uptime_hours=0, thermal_throttle_count=0, error_count=0)

    def reset(self) -> bool:
        resp = self._transport.write_read(
            struct.pack("BB", CMD_RESET, 0x00), 2)
        return resp[0] == RESP_OK

    def _encode(self, text: str) -> List[int]:
        """Simple text to token IDs (demo implementation)."""
        tokens = []
        for word in text.split():
            for tid, tword in self._token_map.items():
                if tword.strip().lower() == word.strip().lower():
                    tokens.append(tid)
                    break
            else:
                tokens.append(sum(ord(c) for c in word) % 30000 + 1000)
        return tokens if tokens else [29892]

    def _decode_token(self, token_id: int) -> str:
        return self._token_map.get(token_id, f"[{token_id}]")

    def benchmark(self, num_prompts: int = 5, max_tokens: int = 64) -> Dict:
        """Run latency/throughput benchmark."""
        prompts = ["Hello", "The meaning of life", "Write code", "Explain AI", "Tell a story"]
        latencies = []
        throughputs = []
        total_tokens = 0

        for i in range(num_prompts):
            prompt = prompts[i % len(prompts)]
            result = self.generate(prompt, max_tokens=max_tokens, stream=True)
            latencies.append(result.elapsed_ms)
            throughputs.append(result.tokens_per_second)
            total_tokens += result.generated_tokens

        return {
            "num_prompts": num_prompts,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "min_latency_ms": round(min(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "avg_tok_per_sec": round(sum(throughputs) / len(throughputs), 1),
            "max_tok_per_sec": round(max(throughputs), 1),
            "total_tokens": total_tokens,
        }

    def fleet_status(self, chips: List["FrozenIntelligenceSDK"]) -> Dict:
        """Report status of multiple chips in a fleet."""
        statuses = []
        for chip in chips:
            h = chip.health()
            statuses.append({"port": chip.port, **h.to_dict()})
        return {"chips": len(chips), "statuses": statuses}


def demo():
    print("=== Frozen Intelligence SDK ===\n")

    # Create SDK instance (simulated)
    sdk = FrozenIntelligenceSDK()

    # Connect and identify
    print("--- Connection ---")
    connected = sdk.connect()
    print(f"  Connected: {connected}")

    info = sdk.identify()
    if info:
        print(f"  Vessel: {info.vessel_class.value}")
        print(f"  Model: {info.model_name}")
        print(f"  Firmware: {info.firmware_version}")
        print(f"  Serial: {info.serial_number}")
        print(f"  Context: {info.max_context} tokens")
        print(f"  Clock: {info.clock_mhz}MHz @ {info.process_nm}nm INT{info.precision_bits}")
    print()

    # Generate with streaming
    print("--- Streaming Generation ---")
    token_count = [0]
    def on_tok(tid, text):
        token_count[0] += 1
        if text.strip():
            print(text, end="", flush=True)

    sdk.on_token(on_tok)
    result = sdk.generate("Hello", max_tokens=8, temperature=0.7, stream=True)
    print(f"\n  Tokens: {result.generated_tokens} | Time: {result.elapsed_ms:.0f}ms | "
          f"Speed: {result.tokens_per_second:.0f} tok/s")
    print()

    # Health check
    print("--- Health ---")
    h = sdk.health()
    for k, v in h.to_dict().items():
        print(f"  {k}: {v}")
    print()

    # Benchmark
    print("--- Benchmark (5 prompts, 64 tokens each) ---")
    bench = sdk.benchmark(5, 64)
    print(f"  Avg latency: {bench['avg_latency_ms']}ms")
    print(f"  Avg throughput: {bench['avg_tok_per_sec']} tok/s")
    print(f"  Peak throughput: {bench['max_tok_per_sec']} tok/s")
    print(f"  Total tokens: {bench['total_tokens']}")
    print()

    # Fleet mode
    print("--- Fleet Mode (4 chips) ---")
    classes = [VesselClass.SCOUT, VesselClass.MESSENGER,
               VesselClass.NAVIGATOR, VesselClass.CAPTAIN]
    chips = []
    for i, cls in enumerate(classes):
        info_i = ChipInfo(vessel_class=cls, model_name=f"model-{cls.value}",
                         firmware_version="0.1.0", serial_number=f"FI-{i+1:03d}",
                         max_context=2048, max_tokens=512,
                         clock_mhz=500, process_nm=28, precision_bits=4)
        chips.append(FrozenIntelligenceSDK(
            transport=SimulatedUSBTransport(info_i, latency_ms=5+i*3),
            port=f"/dev/ttyUSB{i}"))

    fleet = sdk.fleet_status(chips)
    for s in fleet["statuses"]:
        print(f"  {s['port']:16s} {s['state']:20s} {s['temperature_c']:5.1f}C {s['power_w']:5.2f}W")


if __name__ == "__main__":
    demo()
