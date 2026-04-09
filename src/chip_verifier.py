#!/usr/bin/env python3
"""Chip verification suite — formal checks, test pattern generation, and timing analysis."""
import json, math, hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class CheckResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


class CheckSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


@dataclass
class CheckItem:
    name: str
    result: CheckResult
    severity: CheckSeverity
    details: str = ""
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


class TimingAnalyzer:
    """Analyze timing constraints for the mask-locked chip."""

    def __init__(self, process_nm: int = 28, clock_mhz: int = 500):
        self.process = process_nm
        self.clock_mhz = clock_mhz
        self.clock_period_ns = 1000.0 / clock_mhz
        # Library timing estimates (ns) for different processes
        self.lib = {
            28: {"combo_ns": 0.3, "setup_ns": 0.15, "hold_ns": 0.05, "clk_q_ns": 0.1},
            40: {"combo_ns": 0.5, "setup_ns": 0.25, "hold_ns": 0.08, "clk_q_ns": 0.15},
            65: {"combo_ns": 1.0, "setup_ns": 0.5, "hold_ns": 0.15, "clk_q_ns": 0.3},
        }.get(process_nm, {28: {"combo_ns": 0.3, "setup_ns": 0.15, "hold_ns": 0.05, "clk_q_ns": 0.1}})

    def check_timing(self, logic_depth: int) -> CheckItem:
        """Check if logic depth fits within clock period."""
        t = self.lib
        total_ns = t["clk_q_ns"] + logic_depth * t["combo_ns"] + t["setup_ns"]
        slack_ns = self.clock_period_ns - total_ns
        slack_pct = (slack_ns / self.clock_period_ns) * 100

        if slack_ns < 0:
            return CheckItem(f"timing_depth_{logic_depth}", CheckResult.FAIL, CheckSeverity.CRITICAL,
                           f"{logic_depth} levels, {total_ns:.3f}ns > {self.clock_period_ns:.3f}ns period. SLACK: {slack_ns:.3f}ns",
                           slack_ns, 0)
        elif slack_pct < 10:
            return CheckItem(f"timing_depth_{logic_depth}", CheckResult.WARN, CheckSeverity.MAJOR,
                           f"{logic_depth} levels, {total_ns:.3f}ns / {self.clock_period_ns:.3f}ns. Slack: {slack_ns:.3f}ns ({slack_pct:.1f}%)",
                           slack_ns, 0.1 * self.clock_period_ns)
        else:
            return CheckItem(f"timing_depth_{logic_depth}", CheckResult.PASS, CheckSeverity.INFO,
                           f"{logic_depth} levels, {total_ns:.3f}ns / {self.clock_period_ns:.3f}ns. Slack: {slack_ns:.3f}ns ({slack_pct:.1f}%)",
                           slack_ns, 0)

    def max_logic_depth(self) -> int:
        """Calculate maximum logic levels that fit in one clock cycle."""
        t = self.lib
        budget = self.clock_period_ns - t["clk_q_ns"] - t["setup_ns"]
        return int(budget / t["combo_ns"])

    def estimate_frequency(self, logic_depth: int) -> Dict:
        t = self.lib
        total = t["clk_q_ns"] + logic_depth * t["combo_ns"] + t["setup_ns"]
        max_mhz = 1000.0 / total
        return {"logic_depth": logic_depth, "cycle_ns": round(total, 3),
                "max_freq_mhz": round(max_mhz, 0), "utilization_pct": round(self.clock_mhz / max_mhz * 100, 1)}


class PowerAnalyzer:
    """Power analysis for mask-locked chip."""

    def __init__(self, process_nm: int = 28):
        self.process = process_nm
        # mW per mm2 at 1GHz for different activity factors
        self.power_density = {
            28: {"active": 1.2, "idle": 0.05},
            40: {"active": 0.8, "idle": 0.03},
            65: {"active": 0.4, "idle": 0.015},
        }.get(process_nm, {28: {"active": 1.2, "idle": 0.05}})

    def analyze(self, die_area_mm2: float, clock_ghz: float = 0.5,
                activity: float = 0.7, voltage: float = 0.9) -> Dict:
        pd = self.power_density
        # Dynamic power scales with V^2 and frequency
        v_scale = (voltage / 0.9) ** 2
        f_scale = clock_ghz
        dynamic_mw = pd["active"] * die_area_mm2 * activity * f_scale * v_scale
        static_mw = pd["idle"] * die_area_mm2 * 1000  # always-on leakage

        # Mask-locked advantage: no memory access power
        # Traditional chip would add: bandwidth_mw = read_energy * accesses
        memory_savings_pct = 0.35  # 35% of dynamic power is memory access in traditional

        return {
            "die_area_mm2": die_area_mm2,
            "clock_ghz": clock_ghz,
            "activity_factor": activity,
            "voltage_v": voltage,
            "dynamic_mw": round(dynamic_mw, 1),
            "static_mw": round(static_mw, 2),
            "total_mw": round(dynamic_mw + static_mw, 1),
            "watts": round((dynamic_mw + static_mw) / 1000, 3),
            "memory_savings_pct": memory_savings_pct,
            "effective_watts": round((dynamic_mw * (1 - memory_savings_pct) + static_mw) / 1000, 3),
        }


class TestPatternGenerator:
    """Generate test patterns for chip verification."""

    def __init__(self, vocab_size: int = 32000, precision_bits: int = 4):
        self.vocab_size = vocab_size
        self.precision_bits = precision_bits

    def basic_inference_pattern(self) -> List[Dict]:
        """Basic single-token → single-token test."""
        return [
            {"name": "single_token_inference", "steps": [
                {"type": "reset", "cycles": 10},
                {"type": "token", "value": 1, "cycles": 1},
                {"type": "start", "cycles": 1},
                {"type": "wait_valid", "max_cycles": 10000},
                {"type": "check_output", "condition": "token_out != 0"},
                {"type": "wait_done", "max_cycles": 500000},
            ]},
        ]

    def streaming_pattern(self, prompt_tokens: List[int], max_gen: int = 50) -> List[Dict]:
        """Multi-token prompt → streaming generation test."""
        steps = [{"type": "reset", "cycles": 10}]
        steps.append({"type": "config", "temperature": 128, "max_tokens": max_gen})
        steps.append({"type": "new_prompt", "cycles": 1})
        for tok in prompt_tokens:
            steps.append({"type": "token", "value": tok, "cycles": 1})
        steps.append({"type": "start", "cycles": 1})
        steps.append({"type": "wait_valid", "max_cycles": 50000})
        for i in range(max_gen):
            steps.append({"type": "capture_token", "label": f"gen_{i}"})
            steps.append({"type": "wait_valid", "max_cycles": 50000})
        steps.append({"type": "wait_done", "max_cycles": 5000000})
        return [{"name": "streaming_inference", "steps": steps}]

    def edge_case_patterns(self) -> List[Dict]:
        """Corner cases that often catch bugs."""
        return [
            {"name": "empty_prompt", "steps": [
                {"type": "reset", "cycles": 10},
                {"type": "start", "cycles": 1},
                {"type": "wait_done", "max_cycles": 1000},
            ]},
            {"name": "max_vocab_token", "steps": [
                {"type": "reset", "cycles": 10},
                {"type": "token", "value": self.vocab_size - 1, "cycles": 1},
                {"type": "start", "cycles": 1},
                {"type": "wait_done", "max_cycles": 100000},
            ]},
            {"name": "eos_immediate", "steps": [
                {"type": "reset", "cycles": 10},
                {"type": "token", "value": 2, "cycles": 1},  # EOS token
                {"type": "start", "cycles": 1},
                {"type": "wait_done", "max_cycles": 100000},
                {"type": "check", "condition": "gen_count == 0"},
            ]},
            {"name": "thermal_throttle_test", "steps": [
                {"type": "reset", "cycles": 10},
                {"type": "stress", "iterations": 1000},
                {"type": "check", "condition": "max_temp < thermal_limit"},
            ]},
            {"name": "rapid_start_stop", "steps": [
                {"type": "reset", "cycles": 10},
                {"type": "start", "cycles": 1},
                {"type": "start", "cycles": 1},  # double start
                {"type": "wait_done", "max_cycles": 100000},
            ]},
            {"name": "max_context_overflow", "steps": [
                {"type": "reset", "cycles": 10},
                {"type": "token", "value": 42, "cycles": 1},
                {"type": "fill_context", "tokens": 2049},
                {"type": "start", "cycles": 1},
                {"type": "wait_done", "max_cycles": 5000000},
            ]},
        ]

    def generate_weight_checksums(self, layers: Dict[str, List[float]],
                                   precision_bits: int = 4) -> Dict[str, str]:
        """Generate checksums for hardwired weight verification."""
        checksums = {}
        for name, weights in layers.items():
            # In real chip: checksum is computed during tapeout verification
            # Here: hash of quantized weights
            data = "".join(f"{max(-8, min(7, int(w * 8))):04x}" for w in weights[:1000])
            checksums[name] = hashlib.sha256(data.encode()).hexdigest()[:16]
        return checksums


class ChipVerifier:
    """Run all verification checks on a chip design."""

    def __init__(self, process_nm: int = 28, clock_mhz: int = 500):
        self.timing = TimingAnalyzer(process_nm, clock_mhz)
        self.power = PowerAnalyzer(process_nm)
        self.patterns = TestPatternGenerator()
        self.checks: List[CheckItem] = []

    def run_timing_checks(self) -> List[CheckItem]:
        checks = []
        # Check common logic depths
        for depth in [1, 2, 3, 5, 8, 10, 15, 20]:
            checks.append(self.timing.check_timing(depth))

        # Max depth check
        max_d = self.timing.max_logic_depth()
        checks.append(CheckItem("max_logic_depth", CheckResult.PASS, CheckSeverity.INFO,
                               f"Max {max_d} logic levels fit in {self.timing.clock_period_ns:.3f}ns period",
                               float(max_d), None))
        return checks

    def run_power_checks(self, die_area_mm2: float, target_watts: float) -> List[CheckItem]:
        p = self.power.analyze(die_area_mm2)
        effective_w = p["effective_watts"]
        margin = (target_watts - effective_w) / target_watts * 100
        if margin < 0:
            result = CheckResult.FAIL
            sev = CheckSeverity.CRITICAL
        elif margin < 20:
            result = CheckResult.WARN
            sev = CheckSeverity.MAJOR
        else:
            result = CheckResult.PASS
            sev = CheckSeverity.INFO
        return [CheckItem("power_budget", result, sev,
                        f"Effective: {effective_w}W, Target: {target_watts}W, Margin: {margin:.1f}%",
                        effective_w, target_watts)]

    def run_design_rule_checks(self, model_params_b: float, die_area_mm2: float,
                                process_nm: int) -> List[CheckItem]:
        checks = []

        # Die size check (reticle limit ~800mm2 for advanced, ~900mm2 for mature)
        max_die = 900 if process_nm >= 28 else 600
        if die_area_mm2 > max_die:
            checks.append(CheckItem("die_size", CheckResult.FAIL, CheckSeverity.CRITICAL,
                                  f"{die_area_mm2}mm2 exceeds {max_die}mm2 reticle limit"))
        elif die_area_mm2 > max_die * 0.8:
            checks.append(CheckItem("die_size", CheckResult.WARN, CheckSeverity.MAJOR,
                                  f"{die_area_mm2}mm2 close to {max_die}mm2 reticle limit"))
        else:
            checks.append(CheckItem("die_size", CheckResult.PASS, CheckSeverity.INFO,
                                  f"{die_area_mm2}mm2 within {max_die}mm2 reticle limit"))

        # Model fit check
        bits_per_weight = 4
        total_bits = model_params_b * 1e9 * bits_per_weight
        density_mbit_mm2 = {28: 8.0, 40: 4.0, 65: 1.5}.get(process_nm, 4.0)
        min_area = total_bits / (density_mbit_mm2 * 1e6)
        if die_area_mm2 < min_area:
            checks.append(CheckItem("model_fit", CheckResult.FAIL, CheckSeverity.CRITICAL,
                                  f"Need {min_area:.1f}mm2 for weights, have {die_area_mm2}mm2"))
        else:
            utilization = min_area / die_area_mm2 * 100
            checks.append(CheckItem("model_fit", CheckResult.PASS if utilization < 70 else CheckResult.WARN,
                                  CheckSeverity.MAJOR if utilization >= 70 else CheckSeverity.INFO,
                                  f"Weight area: {min_area:.1f}mm2 / {die_area_mm2}mm2 ({utilization:.0f}% utilized)"))

        return checks

    def full_verification(self, model_params_b: float, die_area_mm2: float,
                          target_watts: float) -> Dict:
        self.checks = []
        self.checks.extend(self.run_timing_checks())
        self.checks.extend(self.run_power_checks(die_area_mm2, target_watts))
        self.checks.extend(self.run_design_rule_checks(model_params_b, die_area_mm2, self.timing.process))

        passed = sum(1 for c in self.checks if c.result == CheckResult.PASS)
        failed = sum(1 for c in self.checks if c.result == CheckResult.FAIL)
        warned = sum(1 for c in self.checks if c.result == CheckResult.WARN)
        critical_fails = sum(1 for c in self.checks if c.result == CheckResult.FAIL and c.severity == CheckSeverity.CRITICAL)

        return {
            "summary": {"total": len(self.checks), "passed": passed, "failed": failed,
                       "warned": warned, "critical": critical_fails,
                       "signoff": critical_fails == 0},
            "checks": [{"name": c.name, "result": c.result.value, "severity": c.severity.value,
                        "details": c.details} for c in self.checks],
            "test_patterns": self.patterns.basic_inference_pattern() + self.patterns.edge_case_patterns(),
        }


def demo():
    print("=== Frozen Intelligence: Chip Verification Suite ===\n")

    # Timing analysis
    print("--- Timing Analysis (28nm @ 500MHz) ---")
    timing = TimingAnalyzer(28, 500)
    for depth in [1, 3, 5, 10]:
        c = timing.check_timing(depth)
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}[c.result.value]
        print(f"  {icon} {c.name}: {c.details}")
    print(f"  Max logic depth: {timing.max_logic_depth()} levels")
    print()

    # Power analysis
    print("--- Power Analysis ---")
    power = PowerAnalyzer(28)
    for name, area, target in [("Scout", 100, 1.0), ("Messenger", 595, 3.0),
                                ("Navigator", 1387, 6.0), ("Captain", 2573, 12.0)]:
        p = power.analyze(area)
        margin = (target - p["effective_watts"]) / target * 100
        icon = "✅" if margin > 20 else ("⚠️" if margin > 0 else "❌")
        print(f"  {icon} {name}: {p['effective_watts']}W / {target}W target ({margin:+.0f}% margin)")
    print()

    # Full verification
    print("--- Full Verification: Messenger (3B, 595mm2) ---")
    verifier = ChipVerifier(28, 500)
    report = verifier.full_verification(3.0, 595, 3.0)
    print(f"  Signoff: {'YES ✅' if report['summary']['signoff'] else 'NO ❌'}")
    print(f"  {report['summary']['passed']}/{report['summary']['total']} passed, "
          f"{report['summary']['failed']} failed, {report['summary']['warned']} warnings")
    for c in report["checks"]:
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}[c["result"]]
        print(f"  {icon} [{c['severity']}] {c['name']}: {c['details'][:80]}")

    # Test patterns
    print(f"\n  Test patterns generated: {len(report['test_patterns'])}")
    for tp in report["test_patterns"]:
        print(f"    - {tp['name']}: {len(tp['steps'])} steps")


if __name__ == "__main__":
    demo()
