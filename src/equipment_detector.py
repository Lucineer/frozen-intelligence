#!/usr/bin/env python3
"""Equipment detection — integrates frozen intelligence vessels with deckboss."""
import json, os, sys, subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class DetectionMethod(Enum):
    USB = "usb"
    PCIE = "pcie"
    MOCK = "mock"


@dataclass
class DetectedVessel:
    vessel_id: str
    vessel_class: str
    model_name: str
    interface: str
    serial_number: str
    firmware_version: str
    process_nm: int
    die_area_mm2: float
    max_power_w: float
    target_tok_s: float
    quantization: str
    status: str = "ready"
    temperature_c: float = 25.0
    detection_method: DetectionMethod = DetectionMethod.USB


class VesselScanner:
    """Scans for connected frozen intelligence vessels."""

    USB_VENDOR_ID = "0xFROZ"  # hypothetical vendor ID
    KNOWN_MODELS = {
        "FI-SCOUT-001": {"class": "scout", "model": "qwen3.5-1b-int2", "params": "1B", "tok_s": 100, "power": 0.8},
        "FI-MESSENGER-001": {"class": "messenger", "model": "qwen3.5-3b-int4", "params": "3B", "tok_s": 80, "power": 2.5},
        "FI-NAVIGATOR-001": {"class": "navigator", "model": "qwen3.5-7b-int4", "params": "7B", "tok_s": 50, "power": 5.0},
        "FI-CAPTAIN-001": {"class": "captain", "model": "qwen3.5-13b-int4", "params": "13B", "tok_s": 30, "power": 10.0},
    }

    def scan_usb(self) -> List[DetectedVessel]:
        """Scan USB bus for frozen intelligence vessels."""
        vessels = []
        try:
            result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split("\n"):
                if "Frozen" in line or "FROZ" in line:
                    vessels.append(self._parse_usb_line(line))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return vessels

    def scan_mock(self) -> List[DetectedVessel]:
        """Return mock vessels for testing."""
        vessels = []
        for model_id, spec in self.KNOWN_MODELS.items():
            vessels.append(DetectedVessel(
                vessel_id=f"fi-{spec['class']}-001",
                vessel_class=spec["class"],
                model_name=spec["model"],
                interface="usb3",
                serial_number=f"SN-{model_id}-DEMO",
                firmware_version="1.0.0",
                process_nm=28,
                die_area_mm2=round(4.0 + float(spec["params"].rstrip("B")) * 0.5, 1),
                max_power_w=spec["power"],
                target_tok_s=spec["tok_s"],
                quantization=spec["model"].split("-")[-1],
                detection_method=DetectionMethod.MOCK))
        return vessels

    def scan(self, mock: bool = False) -> List[DetectedVessel]:
        vessels = []
        if mock:
            vessels.extend(self.scan_mock())
        vessels.extend(self.scan_usb())
        return vessels

    def _parse_usb_line(self, line: str) -> DetectedVessel:
        # Real USB parsing would extract vendor/product/serial
        return DetectedVessel(
            vessel_id="fi-unknown-usb",
            vessel_class="unknown",
            model_name="unknown",
            interface="usb3",
            serial_number="USB-UNKNOWN",
            firmware_version="0.0.0",
            process_nm=0,
            die_area_mm2=0,
            max_power_w=0,
            target_tok_s=0,
            quantization="unknown",
            detection_method=DetectionMethod.USB)

    def detect_character_sheet_entry(self, vessel: DetectedVessel) -> Dict:
        """Generate a character sheet entry for a detected vessel."""
        return {
            "equipment_type": "frozen-intelligence",
            "vessel_class": vessel.vessel_class,
            "model": vessel.model_name,
            "interface": vessel.interface,
            "capabilities": {
                "inference": True,
                "max_tokens_per_request": 2048,
                "streaming": True,
                "kv_cache_size_mb": {
                    "scout": 4, "messenger": 16, "navigator": 32, "captain": 64
                }.get(vessel.vessel_class, 16),
            },
            "constraints": {
                "model_frozen": True,
                "no_finetuning": True,
                "stateless_between_requests": True,
                "thermal_throttle_c": 85.0,
            },
            "performance": {
                "tokens_per_second": vessel.target_tok_s,
                "power_watts": vessel.max_power_w,
                "quantization": vessel.quantization,
                "process_nm": vessel.process_nm,
            }
        }


class DeckbossIntegration:
    """Integrates frozen intelligence with deckboss CLI."""

    def __init__(self):
        self.scanner = VesselScanner()
        self.detected: List[DetectedVessel] = []

    def scan_equipment(self, mock: bool = True) -> Dict:
        self.detected = self.scanner.scan(mock=mock)
        if not self.detected:
            return {
                "status": "no_vessels_found",
                "message": "No frozen intelligence vessels detected. Connect a vessel and run again.",
                "character_sheet_updates": []
            }

        updates = []
        for vessel in self.detected:
            entry = self.scanner.detect_character_sheet_entry(vessel)
            entry["vessel_id"] = vessel.vessel_id
            entry["serial"] = vessel.serial_number
            entry["firmware"] = vessel.firmware_version
            updates.append(entry)

        return {
            "status": "found",
            "vessel_count": len(self.detected),
            "vessels": updates,
        }

    def generate_character_sheet_patch(self) -> str:
        """Generate JSON patch for character sheet."""
        equip = self.scan_equipment(mock=True)
        lines = ["# Frozen Intelligence Equipment Detected\n"]
        for v in equip.get("vessels", []):
            lines.append(f"## {v['vessel_id']} ({v['vessel_class']})")
            lines.append(f"- Model: {v['model']}")
            lines.append(f"- Performance: {v['performance']['tokens_per_second']} tok/s @ {v['performance']['power_watts']}W")
            lines.append(f"- Quantization: {v['performance']['quantization']} on {v['performance']['process_nm']}nm")
            lines.append(f"- Interface: {v['interface']}")
            lines.append(f"- Status: ready")
            lines.append("")
        return "\n".join(lines)


def demo():
    print("=== Frozen Intelligence: Equipment Detection ===\n")

    integration = DeckbossIntegration()

    # Scan for vessels
    result = integration.scan_equipment(mock=True)
    print(f"Scan result: {result['status']}")
    print(f"Vessels found: {result['vessel_count']}\n")

    for vessel in result["vessels"]:
        print(f"  {vessel['vessel_id']} ({vessel['vessel_class']})")
        print(f"    Model: {vessel['model']}")
        print(f"    Perf: {vessel['performance']['tokens_per_second']} tok/s @ {vessel['performance']['power_watts']}W")
        print(f"    Quant: {vessel['performance']['quantization']} @ {vessel['performance']['process_nm']}nm")
        print(f"    Serial: {vessel['serial']}")
        print(f"    Firmware: {vessel['firmware']}")
        print()

    # Character sheet patch
    print("--- Character Sheet Patch ---")
    print(integration.generate_character_sheet_patch())


if __name__ == "__main__":
    demo()
