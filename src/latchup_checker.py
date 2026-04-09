#!/usr/bin/env python3
"""Latch-up susceptibility checker for CMOS chips.

Analyzes well spacing, guard rings, I/O protection, and
substrate resistance for latch-up risk assessment.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class WellPair:
    name: str
    nwell_to_psub_um: float    # N-well to P-substrate spacing
    has_guard_ring: bool
    has_substrate_tap: bool
    io_cell: bool = False
    pplus_to_nwell_um: float = 0
    nplus_to_pwell_um: float = 0


@dataclass
class LatchUpResult:
    pair: WellPair
    risk_level: str       # LOW, MEDIUM, HIGH, CRITICAL
    trigger_current_ma: float
    holding_voltage_v: float
    recommendations: List[str]


class LatchUpChecker:
    """CMOS latch-up susceptibility analysis."""

    def __init__(self, process_nm: int = 28):
        self.nm = process_nm

        # Minimum well spacing rules per process (um)
        self.spacing_rules = {
            28: {
                "nwell_to_psub_min": 0.4,
                "nwell_to_psub_safe": 1.0,
                "io_nwell_to_psub_safe": 2.0,
                "guard_ring_width": 0.3,
                "guard_ring_spacing": 0.5,
                "substrate_tap_spacing": 5.0,
                "pplus_to_nwell_min": 0.15,
                "nplus_to_pwell_min": 0.15,
            },
            40: {
                "nwell_to_psub_min": 0.5,
                "nwell_to_psub_safe": 1.2,
                "io_nwell_to_psub_safe": 2.5,
                "guard_ring_width": 0.4,
                "guard_ring_spacing": 0.6,
                "substrate_tap_spacing": 6.0,
                "pplus_to_nwell_min": 0.2,
                "nplus_to_pwell_min": 0.2,
            },
            65: {
                "nwell_to_psub_min": 0.8,
                "nwell_to_psub_safe": 1.5,
                "io_nwell_to_psub_safe": 3.0,
                "guard_ring_width": 0.5,
                "guard_ring_spacing": 0.8,
                "substrate_tap_spacing": 8.0,
                "pplus_to_nwell_min": 0.3,
                "nplus_to_pwell_min": 0.3,
            },
        }
        self.rules = self.spacing_rules.get(process_nm, self.spacing_rules[28])

        # Substrate resistance (ohm/sq) — lower = better for latch-up
        self.substrate_rho = {28: 0.1, 40: 0.08, 65: 0.05, 130: 0.03}

        # PNP/NPN beta products for trigger current estimation
        self.rnpn_base = 10.0   # lateral NPN beta
        self.rpnp_base = 20.0   # vertical PNP beta

    def check(self, pair: WellPair) -> LatchUpResult:
        """Check a well pair for latch-up susceptibility."""
        rules = self.rules
        recs = []

        # Spacing check
        if pair.io_cell:
            safe_spacing = rules["io_nwell_to_psub_safe"]
            spacing_ok = pair.nwell_to_psub_um >= safe_spacing
        else:
            safe_spacing = rules["nwell_to_psub_safe"]
            spacing_ok = pair.nwell_to_psub_um >= safe_spacing

        if not spacing_ok:
            recs.append(f"Increase N-well to P-sub spacing: "
                       f"{pair.nwell_to_psub_um:.2f}um < {safe_spacing:.1f}um safe")

        # Guard ring check
        if not pair.has_guard_ring:
            recs.append("Add P+ guard ring around N-well")

        # Substrate tap check
        if not pair.has_substrate_tap:
            recs.append(f"Add P+ substrate tap within {rules['substrate_tap_spacing']}um")

        # Compute trigger current (higher = safer)
        spacing_factor = pair.nwell_to_psub_um / safe_spacing
        guard_factor = 2.0 if pair.has_guard_ring else 1.0
        tap_factor = 1.5 if pair.has_substrate_tap else 1.0
        io_factor = 0.5 if pair.io_cell else 1.0

        base_trigger = 50.0  # mA base trigger current
        trigger_ma = base_trigger * spacing_factor * guard_factor * tap_factor * io_factor

        # Holding voltage estimation
        beta_product = self.rnpn_base * self.rpnp_base * guard_factor
        holding_v = 0.5 + (1.0 / (1 + beta_product / 100))

        # Risk assessment
        if spacing_ok and pair.has_guard_ring and pair.has_substrate_tap:
            risk = "LOW"
        elif spacing_ok and (pair.has_guard_ring or pair.has_substrate_tap):
            risk = "MEDIUM"
        elif not spacing_ok and pair.io_cell:
            risk = "CRITICAL"
        elif not spacing_ok:
            risk = "HIGH"
        else:
            risk = "MEDIUM"

        return LatchUpResult(pair, risk, trigger_ma, holding_v, recs)

    def batch_check(self, pairs: List[WellPair]) -> Dict:
        results = [self.check(p) for p in pairs]
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for r in results:
            risk_counts[r.risk_level] += 1

        return {
            "total": len(results),
            "risk_counts": risk_counts,
            "results": results,
            "substrate_rho": self.substrate_rho.get(self.nm, 0.1),
        }

    def io_protection_check(self, n_io: int, has_esd: bool,
                            supply_v: float = 3.3) -> Dict:
        """Check I/O latch-up protection."""
        recs = []
        risk = "LOW"

        if not has_esd:
            recs.append("Add ESD protection diodes on all I/O pads")
            risk = "CRITICAL"

        if supply_v > 1.8 and not has_esd:
            recs.append("High voltage I/O requires double-guard-ring protection")

        # Check I/O density
        max_density = 100 if self.nm <= 28 else 50
        if n_io > max_density:
            recs.append(f"High I/O density ({n_io}): ensure substrate taps every "
                       f"{self.rules['substrate_tap_spacing']}um")

        return {"risk": risk, "n_io": n_io, "esd": has_esd,
                "recommendations": recs}


def demo():
    print("=== Latch-Up Checker ===\n")

    checker = LatchUpChecker(28)
    rules = checker.rules
    print(f"Process: {checker.nm}nm")
    print(f"Min N-well to P-sub: {rules['nwell_to_psub_min']}um (safe: {rules['nwell_to_psub_safe']}um)")
    print(f"IO safe spacing: {rules['io_nwell_to_psub_safe']}um")
    print()

    # Test well pairs
    pairs = [
        WellPair("core_logic_1", 1.2, True, True),
        WellPair("core_logic_2", 0.6, False, False),
        WellPair("io_pad_1", 2.5, True, True, io_cell=True),
        WellPair("io_pad_2", 0.8, False, True, io_cell=True),
        WellPair("analog_block", 1.0, True, False),
        WellPair("dense_std_cell", 0.3, False, False),
        WellPair("memory_periphery", 1.5, True, True),
        WellPair("io_pad_3", 0.5, True, False, io_cell=True),
    ]

    print("--- Well Pair Analysis ---")
    for pair in pairs:
        r = checker.check(pair)
        rec_str = "; ".join(r.recommendations) if r.recommendations else "None"
        print(f"  {pair.name:20s}: {r.risk_level:8s} "
              f"(spacing={pair.nwell_to_psub_um:.1f}um, "
              f"trigger={r.trigger_current_ma:.0f}mA, "
              f"hold={r.holding_voltage_v:.2f}V)")
        if rec_str != "None":
            print(f"    -> {rec_str}")

    # Batch
    print("\n--- Batch Summary ---")
    batch = checker.batch_check(pairs)
    print(f"  Total: {batch['total']}, Risk counts: {batch['risk_counts']}")
    print(f"  Substrate rho: {batch['substrate_rho']} ohm/sq")
    print()

    # I/O protection
    print("--- I/O Protection Check ---")
    io = checker.io_protection_check(48, True, 1.8)
    print(f"  48 I/O with ESD, 1.8V supply: {io['risk']}")
    io2 = checker.io_protection_check(120, False, 3.3)
    print(f"  120 I/O no ESD, 3.3V supply: {io2['risk']}")
    for rec in io2["recommendations"]:
        print(f"    -> {rec}")

    # Process comparison
    print("\n--- Process Comparison (safe spacing rules) ---")
    for nm in [28, 40, 65]:
        c = LatchUpChecker(nm)
        r = c.rules
        print(f"  {nm}nm: min={r['nwell_to_psub_min']}um, "
              f"safe={r['nwell_to_psub_safe']}um, "
              f"io_safe={r['io_nwell_to_psub_safe']}um, "
              f"sub_rho={checker.substrate_rho.get(nm, 0)} ohm/sq")


if __name__ == "__main__":
    demo()
