#!/usr/bin/env python3
"""Comprehensive cost model for mask-locked inference chips.

NRE, unit cost, yield, packaging, testing, and total cost of ownership
across volumes and process nodes.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ProcessCost:
    name: str
    nm: int
    wafer_cost: float      # $ per 300mm wafer
    die_size_mm2: float
    gross_die_per_wafer: int
    defect_density: float   # defects/cm2
    design_nre_k: float     # $K for mask set + design
    mask_set_k: float       # $K for mask set alone
    mpw_price_k: float      # $K for MPW shuttle
    lead_time_weeks: int


# Real-world approximate costs
PROCESSES = {
    28: ProcessCost("TSMC 28LP", 28, 3500, 25, 900, 0.08, 1500, 800, 25, 12),
    40: ProcessCost("TSMC 40LP", 40, 2000, 25, 900, 0.05, 800, 400, 15, 10),
    65: ProcessCost("TSMC 65LP", 65, 1200, 25, 900, 0.03, 400, 200, 8, 8),
    130: ProcessCost("TSMC 130G", 130, 500, 25, 900, 0.015, 150, 80, 3, 6),
    22: ProcessCost("GF 22FDX", 22, 6000, 25, 900, 0.12, 2500, 1500, 40, 14),
    12: ProcessCost("TSMC N12", 12, 12000, 25, 900, 0.2, 5000, 3500, 80, 18),
}


@dataclass
class PackageOption:
    name: str
    cost_per_unit: float    # $
    pins: int
    lead_time_weeks: int
    thermal_resistance: float  # C/W


PACKAGES = {
    "QFN48": PackageOption("QFN48", 0.15, 48, 4, 30),
    "BGA256": PackageOption("BGA256", 0.45, 256, 6, 15),
    "BGA672": PackageOption("BGA672", 1.20, 672, 8, 8),
    "WLCSP": PackageOption("WLCSP", 0.08, 0, 3, 40),
    "FCBGA1024": PackageOption("FCBGA1024", 3.50, 1024, 10, 5),
}


@dataclass
class TestCost:
    name: str
    cost_per_sec: float  # $
    test_time_sec: float
    yield_loss_pct: float  # additional yield loss from testing


TESTS = {
    "wafer_probe": TestCost("Wafer Probe", 0.005, 2.0, 2.0),
    "final_test": TestCost("Final Test", 0.01, 5.0, 1.0),
    "burn_in": TestCost("Burn-in", 0.003, 3600.0, 0.5),
}


def poisson_yield(defect_density: float, die_area_cm2: float) -> float:
    return math.exp(-defect_density * die_area_cm2)


def die_binning(gross: int, good: int) -> Dict[str, float]:
    """Simple binning model."""
    gold = good * 0.85
    silver = good * 0.12
    bronze = good * 0.03
    return {"GOLD": gold, "SILVER": silver, "BRONZE": bronze}


class ChipCostModel:
    """Full chip cost model."""

    def __init__(self, process_nm: int = 28, die_size_mm2: float = 25,
                 package: str = "BGA256"):
        self.process = PROCESSES.get(process_nm, PROCESSES[28])
        self.die_size = die_size_mm2
        self.pkg = PACKAGES.get(package, PACKAGES["BGA256"])
        self.volume: int = 10000
        self.include_burn_in = False

    def wafer_yield(self) -> float:
        die_area_cm2 = self.die_size / 100
        return poisson_yield(self.process.defect_density, die_area_cm2)

    def unit_die_cost(self) -> float:
        good = self.process.gross_die_per_wafer * self.wafer_yield()
        return self.process.wafer_cost / good if good > 0 else float('inf')

    def package_cost(self) -> float:
        return self.pkg.cost_per_unit

    def test_cost_per_unit(self) -> float:
        total = 0
        for name, test in TESTS.items():
            if name == "burn_in" and not self.include_burn_in:
                continue
            total += test.cost_per_sec * test.test_time_sec
        # Account for test yield loss
        test_yield = (1 - TESTS["wafer_probe"].yield_loss_pct / 100) * \
                     (1 - TESTS["final_test"].yield_loss_pct / 100)
        return total / test_yield

    def unit_cost(self) -> Dict:
        die = self.unit_die_cost()
        pkg = self.package_cost()
        test = self.test_cost_per_unit()
        total = die + pkg + test
        return {
            "die": round(die, 3),
            "package": round(pkg, 3),
            "test": round(test, 3),
            "total": round(total, 3),
            "volume": self.volume,
        }

    def nre_amortized(self) -> Dict:
        nre = self.process.design_nre_k * 1000
        mask = self.process.mask_set_k * 1000
        total_nre = nre + mask
        per_unit = total_nre / self.volume if self.volume > 0 else float('inf')
        return {
            "design_nre": nre,
            "mask_set": mask,
            "total_nre": total_nre,
            "per_unit": round(per_unit, 3),
            "break_even_volume": int(total_nre / 1.0) if per_unit > 1 else 0,
        }

    def full_report(self) -> Dict:
        uc = self.unit_cost()
        nre = self.nre_amortized()
        return {
            "process": self.process.name,
            "nm": self.process.nm,
            "die_size_mm2": self.die_size,
            "package": self.pkg.name,
            "wafer_yield_pct": round(self.wafer_yield() * 100, 1),
            "gross_die": self.process.gross_die_per_wafer,
            "good_die": round(self.process.gross_die_per_wafer * self.wafer_yield()),
            "unit_cost": uc,
            "nre": nre,
            "total_cost_per_unit": round(uc["total"] + nre["per_unit"], 3),
            "lead_time_weeks": self.process.lead_time_weeks + self.pkg.lead_time_weeks,
        }


def compare_volumes(model: ChipCostModel, volumes: List[int]) -> List[Dict]:
    results = []
    for vol in volumes:
        m = ChipCostModel(model.process.nm, model.die_size, model.pkg.name)
        m.volume = vol
        r = m.full_report()
        r["volume"] = vol
        results.append(r)
    return results


def compare_processes(die_size: float, volume: int) -> List[Dict]:
    results = []
    for nm in sorted(PROCESSES.keys()):
        m = ChipCostModel(nm, die_size)
        m.volume = volume
        r = m.full_report()
        results.append(r)
    return results


def demo():
    print("=== Chip Cost Model ===\n")

    # Base model
    model = ChipCostModel(28, 25, "BGA256")
    report = model.full_report()

    print(f"--- {report['process']}, {report['die_size_mm2']}mm2 die, {report['package']} ---")
    print(f"  Wafer yield: {report['wafer_yield_pct']}%")
    print(f"  Die: {report['gross_die']} gross, {report['good_die']} good per wafer")
    uc = report["unit_cost"]
    print(f"  Unit cost: die=${uc['die']:.2f} + pkg=${uc['package']:.2f} + test=${uc['test']:.2f} = ${uc['total']:.2f}")
    nre = report["nre"]
    print(f"  NRE: design=${nre['design_nre']/1000:.0f}K + mask=${nre['mask_set']/1000:.0f}K = ${nre['total_nre']/1000:.0f}K")
    print(f"  Total (incl NRE @10K): ${report['total_cost_per_unit']:.2f}")
    print(f"  Break-even: {nre['break_even_volume']:,} units")
    print(f"  Lead time: {report['lead_time_weeks']} weeks")
    print()

    # Volume comparison
    print("--- Volume Sensitivity (28nm, 25mm2) ---")
    for vol in [100, 1000, 10000, 100000, 1000000]:
        m = ChipCostModel(28, 25)
        m.volume = vol
        r = m.full_report()
        total = r["total_cost_per_unit"]
        print(f"  {vol:>10,}: ${total:.2f}/unit (die=${r['unit_cost']['die']:.2f}, NRE=${r['nre']['per_unit']:.2f})")
    print()

    # Process comparison
    print("--- Process Comparison (25mm2, 10K units) ---")
    for nm in sorted(PROCESSES.keys()):
        m = ChipCostModel(nm, 25)
        m.volume = 10000
        r = m.full_report()
        print(f"  {nm:3d}nm: ${r['total_cost_per_unit']:.2f}/unit, "
              f"yield={r['wafer_yield_pct']:.1f}%, lead={r['lead_time_weeks']}wk")
    print()

    # Die size impact
    print("--- Die Size Impact (28nm, 10K) ---")
    for size in [4, 9, 16, 25, 36, 49, 64, 100]:
        m = ChipCostModel(28, size)
        m.volume = 10000
        r = m.full_report()
        print(f"  {size:3d}mm2: ${r['total_cost_per_unit']:.2f}/unit, yield={r['wafer_yield_pct']:.1f}%")
    print()

    # MPW option
    print("--- MPW vs Full Mask ---")
    mpw_cost = PROCESSES[28].mpw_price_k * 1000
    full_cost = PROCESSES[28].design_nre_k * 1000
    print(f"  MPW shuttle: ${mpw_cost/1000:.0f}K (5 dies)")
    print(f"  Full mask set: ${full_cost/1000:.0f}K")
    print(f"  Savings: ${(full_cost - mpw_cost)/1000:.0f}K ({(1 - mpw_cost/full_cost)*100:.0f}%)")


if __name__ == "__main__":
    demo()
