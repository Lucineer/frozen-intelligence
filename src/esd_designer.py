#!/usr/bin/env python3
"""ESD protection circuit designer for mask-locked chips.

I/O ESD clamp sizing, diode protection, HBM/CDM/MM model analysis,
and protection level selection per I/O type.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ESDClamp:
    name: str
    esd_type: str       # HBM, CDM, MM
    target_voltage_kv: float  # target protection level
    clamp_type: str     # ggNMOS, diode, scr, gcNMOS
    width_um: float
    on_resistance_ohm: float
    trigger_voltage_v: float
    holding_voltage_v: float
    leakage_na: float


@dataclass
class IOPad:
    name: str
    io_type: str        # input, output, bidir, analog, power
    voltage_domain: float  # VDD
    max_signal_v: float
    has_protection: bool = False
    clamp: Optional[ESDClamp] = None


@dataclass
class ESDResult:
    pad: IOPad
    hbm_kv: float
    cdm_kv: float
    mm_kv: float
    meets_target: bool
    recommendations: List[str]


class ESDDesigner:
    """ESD protection design and analysis."""

    def __init__(self, process_nm: int = 28):
        self.nm = process_nm
        self.clamps: List[ESDClamp] = []

        # Standard ESD targets per domain
        self.targets = {
            "1.8": {"hbm_kv": 2.0, "cdm_kv": 0.5, "mm_kv": 0.2},
            "1.2": {"hbm_kv": 1.5, "cdm_kv": 0.375, "mm_kv": 0.15},
            "3.3": {"hbm_kv": 4.0, "cdm_kv": 1.0, "mm_kv": 0.4},
            "0.9": {"hbm_kv": 1.0, "cdm_kv": 0.25, "mm_kv": 0.1},
        }

        # Process-specific clamp parameters
        self.clamp_params = {
            28: {
                "ggNMOS": {"r_per_um": 2.0, "trigger_v": 1.2, "hold_v": 0.6,
                          "leak_per_um": 0.5},
                "diode": {"r_per_um": 1.0, "trigger_v": 0.6, "hold_v": 0.5,
                         "leak_per_um": 0.1},
                "scr": {"r_per_um": 0.5, "trigger_v": 1.5, "hold_v": 0.3,
                       "leak_per_um": 2.0},
                "gcNMOS": {"r_per_um": 1.5, "trigger_v": 0.8, "hold_v": 0.5,
                          "leak_per_um": 0.3},
            },
        }
        self.params = self.clamp_params.get(process_nm, self.clamp_params[28])

    def design_clamp(self, esd_type: str, target_kv: float,
                     clamp_type: str = "ggNMOS") -> ESDClamp:
        """Size an ESD clamp for target protection level."""
        # HBM: 1.5kohm source, peak current = V/1500
        # CDM: device self-discharge
        # MM: 200ohm source, peak current = V/200
        if esd_type == "HBM":
            peak_current_a = target_kv * 1000 / 1500
        elif esd_type == "MM":
            peak_current_a = target_kv * 1000 / 200
        else:  # CDM
            peak_current_a = target_kv * 1000 / 25  # low impedance

        params = self.params[clamp_type]
        # Size: R_total = V_trigger / peak_current, width = R_per_um * R_total
        width = max(10, peak_current_a / (params["trigger_v"] / params["r_per_um"]))
        width = min(width, 500)  # practical limit
        on_r = params["r_per_um"] / width
        leakage = width * params["leak_per_um"]

        clamp = ESDClamp(
            name=f"{clamp_type}_{esd_type}_{int(target_kv*1000)}V",
            esd_type=esd_type,
            target_voltage_kv=target_kv,
            clamp_type=clamp_type,
            width_um=round(width, 1),
            on_resistance_ohm=round(on_r, 2),
            trigger_voltage_v=params["trigger_v"],
            holding_voltage_v=params["hold_v"],
            leakage_na=round(leakage, 1),
        )
        self.clamps.append(clamp)
        return clamp

    def protect_pad(self, pad: IOPad) -> ESDResult:
        """Design ESD protection for an I/O pad."""
        vdd_str = f"{pad.voltage_domain}"
        targets = self.targets.get(vdd_str, self.targets["1.8"])

        clamp = self.design_clamp("HBM", targets["hbm_kv"], "ggNMOS")
        pad.has_protection = True
        pad.clamp = clamp

        recs = []

        # Additional diode for input pads
        if pad.io_type == "input":
            diode = self.design_clamp("HBM", targets["hbm_kv"], "diode")
            recs.append(f"Add rail-to-rail diode clamp (leak={diode.leakage_na:.0f}nA)")

        # SCR for power pads (highest protection)
        if pad.io_type == "power":
            scr = self.design_clamp("HBM", targets["hbm_kv"] * 1.5, "scr")
            recs.append(f"Add SCR clamp for power pad ({scr.width_um:.0f}um)")

        # Check trigger voltage vs signal
        if pad.max_signal_v > clamp.trigger_voltage_v * 0.8:
            recs.append(f"Warning: signal ({pad.max_signal_v}V) close to "
                       f"trigger ({clamp.trigger_voltage_v}V)")

        return ESDResult(
            pad=pad,
            hbm_kv=targets["hbm_kv"],
            cdm_kv=targets["cdm_kv"],
            mm_kv=targets["mm_kv"],
            meets_target=True,
            recommendations=recs,
        )

    def batch_protect(self, pads: List[IOPad]) -> Dict:
        results = [self.protect_pad(p) for p in pads]
        total_width = sum(r.pad.clamp.width_um for r in results if r.pad.clamp)
        total_leakage = sum(r.pad.clamp.leakage_na for r in results if r.pad.clamp)

        return {
            "total_pads": len(pads),
            "total_clamp_width_um": round(total_width, 0),
            "total_leakage_ua": round(total_leakage / 1000, 2),
            "results": results,
        }


def demo():
    print("=== ESD Protection Designer ===\n")

    esd = ESDDesigner(28)
    print(f"Process: {esd.nm}nm")
    print(f"Targets: {esd.targets}")
    print()

    # Design clamps for each standard
    print("--- Standard Clamp Designs ---")
    for esd_type, kv in [("HBM", 2.0), ("MM", 0.2), ("CDM", 0.5)]:
        for clamp_type in ["ggNMOS", "diode", "scr"]:
            c = esd.design_clamp(esd_type, kv, clamp_type)
            print(f"  {clamp_type:8s} {esd_type:3s} {kv}kV: "
                  f"width={c.width_um:6.1f}um, "
                  f"Ron={c.on_resistance_ohm:.3f}ohm, "
                  f"Vtrig={c.trigger_voltage_v}V, "
                  f"leak={c.leakage_na:.1f}nA")
        print()

    # I/O pad protection
    print("--- I/O Pad Protection ---")
    pads = [
        IOPad("gpio_0", "bidir", 1.8, 1.8),
        IOPad("uart_tx", "output", 1.8, 1.8),
        IOPad("spi_clk", "input", 1.8, 1.8),
        IOPad("vdd_1p8", "power", 1.8, 1.8),
        IOPad("analog_in", "analog", 0.9, 0.9),
        IOPad("vdd_0p9", "power", 0.9, 0.9),
        IOPad("sdio_0", "bidir", 3.3, 3.3),
        IOPad("usb_dp", "bidir", 3.3, 3.3),
    ]

    for pad in pads:
        r = esd.protect_pad(pad)
        status = "PASS" if r.meets_target else "FAIL"
        print(f"  {pad.name:15s} ({pad.io_type:6s}, {pad.voltage_domain}V): "
              f"HBM={r.hbm_kv}kV CDM={r.cdm_kv}kV [{status}]")
        if pad.clamp:
            print(f"    Clamp: {pad.clamp.clamp_type}, "
                  f"{pad.clamp.width_um:.0f}um, "
                  f"Ron={pad.clamp.on_resistance_ohm:.3f}ohm")
        for rec in r.recommendations:
            print(f"    -> {rec}")

    # Batch summary
    print("\n--- Batch Summary ---")
    batch = esd.batch_protect(pads[:4])
    print(f"  {batch['total_pads']} pads, "
          f"total clamp width: {batch['total_clamp_width_um']:.0f}um, "
          f"total leakage: {batch['total_leakage_ua']:.2f}uA")


if __name__ == "__main__":
    demo()
