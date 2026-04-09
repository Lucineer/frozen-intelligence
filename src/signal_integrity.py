#!/usr/bin/env python3
"""Signal integrity analyzer for mask-locked inference chips.

Transmission line modeling, eye diagram generation, crosstalk analysis,
and termination optimization for high-speed interfaces.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


@dataclass
class TransmissionLine:
    """Lossy transmission line model."""
    length_mm: float
    impedance_ohm: float = 50.0
    dielectric_const: float = 4.0
    loss_db_per_mm: float = 0.1  # at 1 GHz

    @property
    def propagation_delay_ps_per_mm(self) -> float:
        return 3.3356 * math.sqrt(self.dielectric_const)  # ps/mm

    def delay_ps(self) -> float:
        return self.length_mm * self.propagation_delay_ps_per_mm

    def loss_at_freq(self, freq_ghz: float) -> float:
        return self.loss_db_per_mm * self.length_mm * math.sqrt(freq_ghz)


@dataclass
class Driver:
    """Output driver model."""
    rise_time_ps: float = 50.0
    output_impedance_ohm: float = 25.0
    voltage_swing_v: float = 0.8  # differential
    max_data_rate_gbps: float = 16.0


@dataclass
class Receiver:
    """Input receiver model."""
    input_impedance_ohm: float = 100.0
    sensitivity_mv: float = 50.0
    setup_time_ps: float = 30.0
    hold_time_ps: float = 10.0


class EyeDiagram:
    """Generate and analyze eye diagram."""

    def __init__(self, tline: TransmissionLine, driver: Driver,
                 receiver: Receiver):
        self.tline = tline
        self.driver = driver
        self.receiver = receiver

    def compute_eye(self, data_rate_gbps: float,
                    n_bits: int = 1000) -> Dict:
        """Compute eye opening."""
        ui_ps = 1000.0 / data_rate_gbps  # unit interval in ps

        # Voltage loss
        loss_db = self.tline.loss_at_freq(data_rate_gbps / 2)
        voltage_ratio = 10 ** (-loss_db / 20)
        swing = self.driver.voltage_swing_v * voltage_ratio

        # Rise time degradation (bandwidth limited)
        bw_ghz = 0.35 / (self.driver.rise_time_ps / 1000)  # 3dB bandwidth
        if bw_ghz < data_rate_gbps / 2:
            risetime_factor = 0.5 * (data_rate_gbps / 2) / bw_ghz
        else:
            risetime_factor = 0.1

        # ISI (inter-symbol interference) closes eye
        isi_closure = min(0.4, 0.1 * math.log2(max(1, data_rate_gbps / 4)))

        # Jitter
        total_jitter_ps = self.driver.rise_time_ps * 0.2 + self.tline.delay_ps() * 0.02
        deterministic_jitter = ui_ps * 0.1
        random_jitter = total_jitter_ps * 0.3

        # Eye opening
        eye_height = swing * (1 - isi_closure - risetime_factor)
        eye_width = ui_ps * (1 - (deterministic_jitter + 2 * random_jitter) / ui_ps)

        # Margin analysis
        voltage_margin = (eye_height / 2 - self.receiver.sensitivity_mv / 1000)
        time_margin = (eye_width / 2 - max(self.receiver.setup_time_ps,
                                             self.receiver.hold_time_ps))

        return {
            "data_rate_gbps": data_rate_gbps,
            "ui_ps": round(ui_ps, 2),
            "loss_db": round(loss_db, 2),
            "swing_v": round(swing, 3),
            "eye_height_mv": round(eye_height * 1000, 1),
            "eye_width_ps": round(eye_width, 1),
            "voltage_margin_mv": round(voltage_margin * 1000, 1),
            "time_margin_ps": round(time_margin, 1),
            "pass": voltage_margin > 0 and time_margin > 0,
            "ber_estimate": round(max(1e-15, 10 ** (-abs(voltage_margin * 1000) / 50)), 2),
        }


class CrosstalkAnalyzer:
    """Near-end and far-end crosstalk analysis."""

    def __init__(self, spacing_um: float = 100.0, coupling_length_mm: float = 5.0,
                 dielectric: float = 4.0):
        self.spacing = spacing_um
        self.coupling_length = coupling_length_mm
        self.dielectric = dielectric

    def next_coupling_db(self, freq_ghz: float) -> float:
        """NEXT (near-end crosstalk) in dB."""
        # Simplified coupling model
        k = (1.0 / self.dielectric) * (0.1 / (self.spacing / 100.0)) ** 2
        next_db = 20 * math.log10(k * self.coupling_length * freq_ghz)
        return max(-60, min(-10, next_db))

    def far_coupling_db(self, freq_ghz: float) -> float:
        """FEXT (far-end crosstalk) in dB."""
        td = self.coupling_length * 3.3356 * math.sqrt(self.dielectric)
        k = (1.0 / self.dielectric) * (0.1 / (self.spacing / 100.0)) ** 2
        fext_db = 20 * math.log10(k * self.coupling_length * freq_ghz) - 20 * math.log10(freq_ghz * 1e9 * td * 1e-12)
        return max(-60, min(-10, fext_db))

    def analyze_spacings(self, freq_ghz: float) -> List[Dict]:
        """Analyze crosstalk across different spacings."""
        results = []
        for spacing in [50, 75, 100, 150, 200, 300]:
            a = CrosstalkAnalyzer(spacing, self.coupling_length, self.dielectric)
            results.append({
                "spacing_um": spacing,
                "next_db": round(a.next_coupling_db(freq_ghz), 1),
                "fext_db": round(a.far_coupling_db(freq_ghz), 1),
            })
        return results


class TerminationOptimizer:
    """Find optimal termination scheme."""

    @staticmethod
    def series_source(zo: float, zs: float, zl: float) -> Dict:
        """Series source termination."""
        rs = zo - zs
        reflection_coeff = abs((zl - zo) / (zl + zo))
        return {"rs_ohm": round(rs, 1), "reflection": round(reflection_coeff, 4),
                "type": "series_source"}

    @staticmethod
    def parallel_receiver(zo: float, zl: float) -> Dict:
        """Parallel receiver termination."""
        rt = zo
        rt_power = zo / 2  # power divider
        return {"rt_ohm": rt, "signal_loss_db": round(20 * math.log10(0.5), 1),
                "type": "parallel_receiver"}

    @staticmethod
    def differential(zo: float) -> Dict:
        """Differential pair termination."""
        return {"rt_ohm": 2 * zo, "cmrr_db": 30, "type": "differential"}


def demo():
    print("=== Signal Integrity Analyzer ===\n")

    tline = TransmissionLine(10.0, 50.0, 4.0, 0.08)
    driver = Driver(50.0, 25.0, 0.8, 16.0)
    receiver = Receiver(100.0, 50.0, 30.0, 10.0)

    # Eye diagram
    print("--- Eye Diagram Analysis ---")
    eye = EyeDiagram(tline, driver, receiver)
    for rate in [1, 4, 8, 16, 25, 32]:
        r = eye.compute_eye(rate)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {rate:2d} Gbps: eye={r['eye_height_mv']:6.1f}mV x {r['eye_width_ps']:5.1f}ps, "
              f"loss={r['loss_db']:.1f}dB, vmarg={r['voltage_margin_mv']:.0f}mV, {status}")
    print()

    # Crosstalk
    print("--- Crosstalk Analysis (4 GHz) ---")
    xt = CrosstalkAnalyzer(100, 5.0, 4.0)
    spacings = xt.analyze_spacings(4.0)
    for s in spacings:
        print(f"  {s['spacing_um']:3d}um: NEXT={s['next_db']:5.1f}dB, FEXT={s['fext_db']:5.1f}dB")
    print()

    # Termination
    print("--- Termination Schemes ---")
    print(f"  Series source (Zo=50): {TerminationOptimizer.series_source(50, 25, 10000)}")
    print(f"  Parallel receiver (Zo=50): {TerminationOptimizer.parallel_receiver(50, 10000)}")
    print(f"  Differential (Zo=50): {TerminationOptimizer.differential(50)}")
    print()

    # Transmission line comparison
    print("--- PCB vs On-Chip Interconnect ---")
    for name, length, dk, loss in [("PCB stripline", 50, 4.0, 0.08),
                                     ("PCB microstrip", 25, 3.5, 0.12),
                                     ("On-chip (BEOL)", 2, 3.9, 0.5),
                                     ("On-chip (MEOL)", 0.5, 3.9, 0.8)]:
        tl = TransmissionLine(length, 50, dk, loss)
        print(f"  {name:20s}: {tl.delay_ps():7.0f}ps delay, "
              f"{tl.loss_at_freq(8):5.1f}dB loss @8GHz")


if __name__ == "__main__":
    demo()
