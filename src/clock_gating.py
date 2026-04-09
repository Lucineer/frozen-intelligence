#!/usr/bin/env python3
"""Clock gating and power management for mask-locked inference chips.

Based on SuperInstance FPGA Prototype Implementation Guide.
Hierarchical clock gating, DVFS (Dynamic Voltage-Frequency Scaling),
thermal throttling, and power estimation.
"""
import time, math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class PowerDomain(Enum):
    CORE = "core"
    MEMORY = "memory"
    IO = "io"
    CONTROL = "control"
    SENSOR = "sensor"


class ClockState(Enum):
    TURBO = ("turbo", 1.0, 300, 5.0)
    NORMAL = ("normal", 0.9, 200, 3.0)
    POWER_SAVE = ("power_save", 0.8, 100, 1.5)
    IDLE = ("idle", 0.7, 10, 0.3)
    OFF = ("off", 0.0, 0, 0.0)

    def __init__(self, label, voltage, freq_mhz, power_w):
        self.label = label
        self.voltage = voltage
        self.freq_mhz = freq_mhz
        self.power_w = power_w


@dataclass
class ClockDomain:
    name: str
    domain: PowerDomain
    base_freq_mhz: int = 500
    gated: bool = False
    activity: float = 0.0  # 0.0 to 1.0
    state: ClockState = ClockState.NORMAL

    @property
    def effective_freq(self) -> float:
        return self.base_freq_mhz * (self.state.freq_mhz / 300) if not self.gated else 0

    @property
    def power(self) -> float:
        if self.gated:
            return 0.001  # leakage
        base = self.state.power_w * (self.base_freq_mhz / 500)
        return base * self.activity


class ClockGatingController:
    """Hierarchical clock gating with activity monitoring."""

    def __init__(self):
        self.domains: Dict[str, ClockDomain] = {}
        self.activity_history: Dict[str, List[float]] = {}
        self.temperature_c = 45.0
        self.power_budget_w = 10.0
        self.throttle_threshold_c = 85.0

    def add_domain(self, name: str, domain: PowerDomain, freq_mhz: int = 500):
        self.domains[name] = ClockDomain(name, domain, freq_mhz)
        self.activity_history[name] = []

    def update_activity(self, name: str, activity: float):
        """Update activity (0.0 to 1.0) for a domain."""
        if name in self.domains:
            self.domains[name].activity = max(0.0, min(1.0, activity))
            self.activity_history[name].append(activity)
            # Keep last 100 samples
            if len(self.activity_history[name]) > 100:
                self.activity_history[name].pop(0)

    def gate_domain(self, name: str, gate: bool = True):
        """Gate (disable) a clock domain."""
        if name in self.domains:
            self.domains[name].gated = gate

    def set_state(self, name: str, state: ClockState):
        """Set DVFS state for a domain."""
        if name in self.domains:
            self.domains[name].state = state

    def auto_gate(self, threshold: float = 0.05):
        """Automatically gate domains with low activity."""
        for name, domain in self.domains.items():
            if domain.activity < threshold and not domain.gated:
                self.gate_domain(name, True)
            elif domain.activity >= threshold and domain.gated:
                self.gate_domain(name, False)

    def thermal_throttle(self, temperature_c: float) -> Dict:
        """Apply thermal throttling based on temperature."""
        self.temperature_c = temperature_c
        actions = {}
        if temperature_c >= self.throttle_threshold_c:
            # Critical: throttle everything
            for name in self.domains:
                if self.domains[name].state != ClockState.POWER_SAVE:
                    old = self.domains[name].state
                    self.set_state(name, ClockState.POWER_SAVE)
                    actions[name] = f"throttle: {old.label} → power_save"
        elif temperature_c >= 75:
            # Warm: throttle non-critical domains
            for name, domain in self.domains.items():
                if domain.domain != PowerDomain.CORE and domain.state != ClockState.NORMAL:
                    old = domain.state
                    self.set_state(name, ClockState.NORMAL)
                    actions[name] = f"throttle: {old.label} → normal"
        return actions

    def total_power(self) -> Dict:
        """Calculate total power by domain."""
        total = 0.0
        by_domain = {}
        for domain in PowerDomain:
            by_domain[domain.value] = 0.0
        for name, dom in self.domains.items():
            pwr = dom.power
            total += pwr
            by_domain[dom.domain.value] += pwr
        return {"total_w": round(total, 2), "by_domain": by_domain,
                "margin_w": round(self.power_budget_w - total, 2)}

    def efficiency_score(self) -> float:
        """Calculate power efficiency score (0.0 to 1.0)."""
        total_activity = sum(d.activity for d in self.domains.values())
        total_power = self.total_power()["total_w"]
        if total_power == 0:
            return 0.0
        # Higher score = more activity per watt
        return round(total_activity / total_power, 3)


class PowerEstimator:
    """Estimate power for mask-locked inference chips."""

    # Power per component (28nm, 500MHz)
    COMPONENT_POWER = {
        "mac_unit": 0.001,      # 1mW per MAC
        "sram_per_mb": 0.05,    # 50mW per MB
        "io_pad": 0.005,        # 5mW per pad
        "pll": 0.1,             # 100mW per PLL
        "sensor": 0.01,         # 10mW per sensor
    }

    @staticmethod
    def estimate_dynamic(mac_count: int, activity: float = 0.5) -> float:
        """Estimate dynamic power for compute."""
        return mac_count * PowerEstimator.COMPONENT_POWER["mac_unit"] * activity

    @staticmethod
    def estimate_static(die_area_mm2: float, process_nm: int = 28) -> float:
        """Estimate static (leakage) power."""
        # Leakage power density (W/mm2) by process
        leakage = {28: 0.01, 40: 0.02, 65: 0.05, 130: 0.1}
        density = leakage.get(process_nm, 0.02)
        return die_area_mm2 * density

    @staticmethod
    def estimate_memory(mb: float, bw_gbs: float = 10.0) -> float:
        """Estimate memory power."""
        sram = mb * PowerEstimator.COMPONENT_POWER["sram_per_mb"]
        # DDR4 power ~1pJ/bit
        ddr = bw_gbs * 1e9 / 8 * 1e-12 * 1e3  # mW
        return sram + ddr / 1000  # W

    @staticmethod
    def full_estimate(model_params_b: float, hidden_dim: int = 768,
                      num_layers: int = 24, process_nm: int = 28) -> Dict:
        """Full power estimate for a chip."""
        # Compute
        mac_per_layer = hidden_dim * hidden_dim * 4  # Q,K,V,O
        mac_total = mac_per_layer * num_layers
        dynamic = PowerEstimator.estimate_dynamic(mac_total, 0.5)

        # Die area estimate (simplified)
        die_mm2 = model_params_b * 4 * 8 * 4 * 1e-9 * 1e6  # 4 F2 per bit
        static = PowerEstimator.estimate_static(die_mm2, process_nm)

        # Memory (on-chip SRAM for KV cache)
        kv_mb = hidden_dim * 2048 * 2 * num_layers * 2 / 8 / 1e6  # 2× context×heads×INT8
        memory = PowerEstimator.estimate_memory(kv_mb)

        # IO (simplified)
        io_pads = 100
        io = io_pads * PowerEstimator.COMPONENT_POWER["io_pad"]

        total = dynamic + static + memory + io
        return {"dynamic_w": round(dynamic, 2), "static_w": round(static, 2),
                "memory_w": round(memory, 2), "io_w": round(io, 2),
                "total_w": round(total, 2), "die_mm2": round(die_mm2, 1)}


def demo():
    print("=== Clock Gating & Power Management ===\n")

    # Clock gating controller
    ctrl = ClockGatingController()
    domains = [
        ("core_array_0", PowerDomain.CORE, 500),
        ("core_array_1", PowerDomain.CORE, 500),
        ("core_array_2", PowerDomain.CORE, 500),
        ("core_array_3", PowerDomain.CORE, 500),
        ("kv_cache", PowerDomain.MEMORY, 300),
        ("weight_bram", PowerDomain.MEMORY, 400),
        ("pcie_if", PowerDomain.IO, 250),
        ("sensor_adc", PowerDomain.SENSOR, 100),
    ]
    for name, domain, freq in domains:
        ctrl.add_domain(name, domain, freq)

    print("--- Initial State ---")
    for name in ["core_array_0", "kv_cache", "pcie_if"]:
        ctrl.update_activity(name, 0.8)
    ctrl.update_activity("sensor_adc", 0.1)
    ctrl.auto_gate(0.1)
    power = ctrl.total_power()
    print(f"  Active domains: {sum(1 for d in ctrl.domains.values() if not d.gated)}/{len(domains)}")
    print(f"  Total power: {power['total_w']}W (budget: {ctrl.power_budget_w}W)")
    print(f"  Efficiency: {ctrl.efficiency_score()} activity/W")
    print()

    # Thermal throttling scenario
    print("--- Thermal Throttling ---")
    print("  Temperature: 45°C → normal operation")
    actions = ctrl.thermal_throttle(45)
    if actions:
        for a in actions.values():
            print(f"    {a}")
    print("  Temperature: 80°C → throttling")
    actions = ctrl.thermal_throttle(80)
    for a in actions.values():
        print(f"    {a}")
    power2 = ctrl.total_power()
    print(f"  Power after throttle: {power2['total_w']}W (saved {power['total_w'] - power2['total_w']:.1f}W)")
    print()

    # DVFS states
    print("--- DVFS States ---")
    for state in ClockState:
        print(f"  {state.label:12s}: {state.freq_mhz:4d}MHz, {state.voltage:4.1f}V, {state.power_w:5.1f}W")
    print()

    # Power estimation
    print("--- Power Estimation (28nm process) ---")
    est = PowerEstimator()
    for name, params, hd, layers in [
        ("Scout", 1.0, 512, 12),
        ("Messenger", 3.0, 768, 24),
        ("Navigator", 7.0, 1024, 32),
        ("Captain", 13.0, 1536, 40),
    ]:
        p = est.full_estimate(params, hd, layers, 28)
        print(f"  {name:12s} ({params}B): {p['total_w']:5.1f}W total")
        print(f"    Dynamic: {p['dynamic_w']:4.1f}W, Static: {p['static_w']:4.1f}W, "
              f"Memory: {p['memory_w']:4.1f}W, Die: {p['die_mm2']:5.0f}mm2")
    print()

    # Power savings from clock gating
    print("--- Power Savings Analysis ---")
    print("  Without clock gating (all domains active @ 50% activity):")
    for name in ctrl.domains:
        ctrl.gate_domain(name, False)
        ctrl.update_activity(name, 0.5)
    p1 = ctrl.total_power()
    print(f"    Power: {p1['total_w']}W")
    print("  With clock gating (auto-gate <10% activity):")
    ctrl.auto_gate(0.1)
    p2 = ctrl.total_power()
    print(f"    Power: {p2['total_w']}W")
    print(f"  Savings: {p1['total_w'] - p2['total_w']:.1f}W ({((p1['total_w'] - p2['total_w']) / p1['total_w'] * 100):.0f}%)")


if __name__ == "__main__":
    demo()
