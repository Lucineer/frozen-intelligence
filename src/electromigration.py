#!/usr/bin/env python3
"""Electromigration (EM) analyzer for mask-locked chip interconnects.

Black's equation for wire lifetime, current density limits,
and per-layer EM checking.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class WireSegment:
    name: str
    layer: str          # M1, M2, etc
    width_um: float
    length_um: float
    thickness_um: float
    current_ma: float   # average current (mA)
    duty_cycle: float   # 0-1 signal activity
    temperature_c: float

    @property
    def cross_section_um2(self) -> float:
        return self.width_um * self.thickness_um

    @property
    def current_density_ma_per_um2(self) -> float:
        return self.current_ma / max(self.cross_section_um2, 1e-6)

    @property
    def rms_current_ma(self) -> float:
        return self.current_ma * math.sqrt(self.duty_cycle)


@dataclass
class EMResult:
    segment: WireSegment
    mttf_hours: float        # mean time to failure
    meets_spec: bool
    max_allowed_current_ma: float
    headroom_pct: float


class EMAnalyzer:
    """Electromigration analysis using Black's equation."""

    def __init__(self, process_nm: int = 28):
        self.nm = process_nm

        # Black's equation parameters per process
        # MTTF = A * (J - J_crit)^-n * exp(Ea / kT)
        # A is process-dependent constant, Ea = activation energy
        self.ea = 0.7  # eV (aluminum) / 0.9 (copper) — use Cu
        self.ea = 0.9  # copper activation energy
        self.k_boltzmann = 8.617e-5  # eV/K
        self.n_exponent = 2.0  # Black's equation exponent (typically 1-2)
        self.j_crit = 0.0  # critical current density (A/cm2), simplified

        # Per-layer current density limits (mA/um2) at 125C reference
        self.dc_limits = {
            28: {"M1": 1.5, "M2": 2.0, "M3": 2.5, "M4": 3.0, "M5": 3.5, "M6": 4.0},
            40: {"M1": 1.8, "M2": 2.5, "M3": 3.0, "M4": 3.5, "M5": 4.0},
            65: {"M1": 2.0, "M2": 3.0, "M3": 3.5, "M4": 4.0},
            130: {"M1": 2.5, "M2": 3.5, "M3": 4.5},
        }
        self.limits = self.dc_limits.get(process_nm, self.dc_limits[28])

        # Wire dimensions per layer (um)
        self.wire_specs = {
            "M1": {"thickness": 0.2, "min_width": 0.09},
            "M2": {"thickness": 0.22, "min_width": 0.09},
            "M3": {"thickness": 0.25, "min_width": 0.10},
            "M4": {"thickness": 0.30, "min_width": 0.12},
            "M5": {"thickness": 0.40, "min_width": 0.15},
            "M6": {"thickness": 0.50, "min_width": 0.20},
        }

    def blacks_equation(self, j_ma_um2: float, temp_c: float) -> float:
        """Compute MTTF using Black's equation. Returns hours."""
        j_a_cm2 = j_ma_um2 * 1e7  # mA/um2 -> A/cm2
        j_eff = max(j_a_cm2 - self.j_crit, 1e-6)
        t_k = temp_c + 273.15
        # MTTF = A * j^-n * exp(Ea/kT)
        # A chosen so that reference design (1 mA/um2, 125C) = 10 years
        ref_t = 125 + 273.15
        ref_j = 1.0 * 1e7
        ref_mtth = 10 * 365 * 24  # 10 years in hours
        a_const = ref_mtth * (ref_j ** self.n_exponent) / math.exp(self.ea / (self.k_boltzmann * ref_t))

        mttf = a_const * (j_eff ** -self.n_exponent) * math.exp(self.ea / (self.k_boltzmann * t_k))
        return mttf

    def analyze(self, segment: WireSegment, spec_hours: float = 87600,
                safety_factor: float = 10.0) -> EMResult:
        """Analyze a wire segment for EM reliability."""
        j = segment.current_density_ma_per_um2
        limit = self.limits.get(segment.layer, 2.0)

        # Temperature derating
        temp_factor = math.exp(
            (self.ea / self.k_boltzmann) * (1 / (segment.temperature_c + 273.15) - 1 / 398.15))
        derated_limit = limit * temp_factor

        mttf = self.blacks_equation(j, segment.temperature_c)
        target_mttf = spec_hours * safety_factor

        return EMResult(
            segment=segment,
            mttf_hours=mttf,
            meets_spec=mttf >= target_mttf,
            max_allowed_current_ma=derated_limit * segment.cross_section_um2,
            headroom_pct=round((derated_limit - j) / derated_limit * 100, 1) if derated_limit > 0 else 0
        )

    def batch_analyze(self, segments: List[WireSegment]) -> Dict:
        """Analyze multiple segments and report."""
        results = []
        failures = []
        for seg in segments:
            r = self.analyze(seg)
            results.append(r)
            if not r.meets_spec:
                failures.append(r)

        return {
            "total": len(results),
            "pass": len(results) - len(failures),
            "fail": len(failures),
            "worst_mttf_hours": min((r.mttf_hours for r in results), default=0),
            "best_mttf_hours": max((r.mttf_hours for r in results), default=0),
            "failures": [{"name": f.segment.name, "layer": f.segment.layer,
                         "mttf_years": round(f.mttf_hours / 8760, 1),
                         "j_density": round(f.segment.current_density_ma_per_um2, 2)}
                        for f in failures],
        }


def demo():
    print("=== Electromigration Analyzer ===\n")

    em = EMAnalyzer(28)
    print(f"Process: {em.nm}nm, Ea={em.ea}eV, n={em.n_exponent}")
    print(f"DC limits: {em.limits}")
    print()

    # Test wire segments (typical chip interconnects)
    segments = [
        WireSegment("mac_out_bus", "M2", 0.36, 500, 0.22, 12.0, 0.5, 85),
        WireSegment("sram_bitline", "M1", 0.18, 200, 0.20, 0.5, 0.3, 75),
        WireSegment("clock_tree", "M3", 0.30, 3000, 0.25, 25.0, 1.0, 90),
        WireSegment("power_rail_vdd", "M6", 2.00, 5000, 0.50, 800.0, 1.0, 95),
        WireSegment("data_bus", "M4", 0.48, 1000, 0.30, 15.0, 0.4, 80),
        WireSegment("signal_short", "M1", 0.09, 5, 0.20, 0.1, 0.2, 70),
        WireSegment("power_rail_gnd", "M6", 2.00, 5000, 0.50, 800.0, 1.0, 95),
        WireSegment("dense_signal", "M2", 0.09, 50, 0.22, 2.0, 0.6, 85),
    ]

    # Individual analysis
    print("--- Wire Segment Analysis ---")
    for seg in segments:
        r = em.analyze(seg)
        j = seg.current_density_ma_per_um2
        limit = em.limits.get(seg.layer, 2.0)
        status = "PASS" if r.meets_spec else "FAIL"
        print(f"  {seg.name:20s} {seg.layer}: J={j:.2f}mA/um2 "
              f"(limit={limit:.1f}), headroom={r.headroom_pct:+.0f}%, "
              f"MTTF={r.mttf_hours/8760:.0f}yr [{status}]")

    # Batch
    print("\n--- Batch Summary ---")
    batch = em.batch_analyze(segments)
    print(f"  Total: {batch['total']}, Pass: {batch['pass']}, Fail: {batch['fail']}")
    print(f"  MTTF range: {batch['worst_mttf_hours']/8760:.0f}yr - {batch['best_mttf_hours']/8760:.0f}yr")
    if batch["failures"]:
        print("  Failures:")
        for f in batch["failures"]:
            print(f"    {f['name']} ({f['layer']}): J={f['j_density']}mA/um2, MTTF={f['mttf_years']}yr")
    print()

    # Temperature sensitivity
    print("--- MTTF vs Temperature (M2, 1mA/um2, DC) ---")
    for temp in [25, 50, 75, 100, 125, 150]:
        mttf = em.blacks_equation(1.0, temp)
        print(f"  {temp:3d}C: MTTF = {mttf/8760:.0f} years")

    # Current density limits vs temperature
    print("\n--- Derated DC Limits vs Temperature (M2) ---")
    base_limit = em.limits.get("M2", 2.0)
    for temp in [25, 50, 75, 100, 125]:
        tf = math.exp((em.ea / em.k_boltzmann) * (1 / (temp + 273.15) - 1 / 398.15))
        derated = base_limit * tf
        print(f"  {temp:3d}C: {derated:.2f} mA/um2 (base: {base_limit:.1f})")


if __name__ == "__main__":
    demo()
