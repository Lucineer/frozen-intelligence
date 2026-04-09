#!/usr/bin/env python3
"""Design Rule Checker (DRC) for mask-locked inference chips.

Checks geometric layout rules: minimum width, spacing, enclosure, density.
Based on TSMC 28nm LP design rules (simplified).
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class DRCSeverity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class DRCCategory(Enum):
    WIDTH = "Minimum Width"
    SPACING = "Minimum Spacing"
    ENCLOSURE = "Minimum Enclosure"
    DENSITY = "Metal Density"
    EXTENSION = "Minimum Extension"
    OVERLAP = "Required Overlap"


@dataclass
class DRCRule:
    category: DRCCategory
    layer: int
    value: float  # micrometers
    description: str
    severity: DRCSeverity = DRCSeverity.ERROR


@dataclass
class DRCViolation:
    rule: DRCRule
    x: float
    y: float
    actual: float
    expected: float
    shape_name: str = ""


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    layer: int
    name: str = ""

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def perimeter(self) -> float:
        return 2 * (self.w + self.h)

    def overlaps(self, other: 'Rect') -> bool:
        return not (self.x + self.w <= other.x or other.x + other.w <= self.x or
                    self.y + self.h <= other.y or other.y + other.h <= self.y)

    def spacing_to(self, other: 'Rect') -> float:
        if self.overlaps(other):
            return 0.0
        dx = max(other.x - (self.x + self.w), self.x - (other.x + other.w), 0)
        dy = max(other.y - (self.y + self.h), self.y - (other.y + other.h), 0)
        return math.sqrt(dx * dx + dy * dy)

    def contains(self, other: 'Rect') -> bool:
        return (self.x <= other.x and self.y <= other.y and
                self.x + self.w >= other.x + other.w and
                self.y + self.h >= other.y + other.h)

    def enclosure(self, other: 'Rect') -> Tuple[float, float]:
        """Left/bottom and right/top enclosure."""
        lb = min(other.x - self.x, other.y - self.y)
        rt = min(self.x + self.w - (other.x + other.w),
                 self.y + self.h - (other.y + other.h))
        return (lb, rt)


class DesignRuleChecker:
    """Check layout against design rules."""

    def __init__(self, process_nm: int = 28):
        self.process_nm = process_nm
        self.rules = self._default_rules()
        self.violations: List[DRCViolation] = []

    def _default_rules(self) -> List[DRCRule]:
        """Simplified TSMC 28nm LP rules."""
        rules = [
            # Minimum widths
            DRCRule(DRCCategory.WIDTH, 1, 0.065, "Poly gate minimum width"),
            DRCRule(DRCCategory.WIDTH, 2, 0.090, "Metal1 minimum width"),
            DRCRule(DRCCategory.WIDTH, 3, 0.100, "Metal2 minimum width"),
            DRCRule(DRCCategory.WIDTH, 4, 0.070, "Via1 minimum size"),
            DRCRule(DRCCategory.WIDTH, 5, 0.070, "Contact minimum size"),
            DRCRule(DRCCategory.WIDTH, 6, 0.090, "Active minimum width"),
            DRCRule(DRCCategory.WIDTH, 9, 25.0, "Pad minimum size"),
            # Minimum spacing
            DRCRule(DRCCategory.SPACING, 1, 0.090, "Poly-to-poly minimum spacing"),
            DRCRule(DRCCategory.SPACING, 2, 0.090, "Metal1-to-Metal1 minimum spacing"),
            DRCRule(DRCCategory.SPACING, 3, 0.100, "Metal2-to-Metal2 minimum spacing"),
            DRCRule(DRCCategory.SPACING, 4, 0.090, "Via1-to-Via1 minimum spacing"),
            DRCRule(DRCCategory.SPACING, 9, 40.0, "Pad-to-Pad minimum spacing"),
            # Minimum enclosure
            DRCRule(DRCCategory.ENCLOSURE, 2, 0.030, "Metal1 over contact enclosure"),
            DRCRule(DRCCategory.ENCLOSURE, 3, 0.030, "Metal2 over Via1 enclosure"),
            DRCRule(DRCCategory.ENCLOSURE, 1, 0.060, "Poly over active enclosure"),
            # Metal density
            DRCRule(DRCCategory.DENSITY, 2, 0.20, "Metal1 minimum density (20%)", DRCSeverity.WARNING),
            DRCRule(DRCCategory.DENSITY, 2, 0.80, "Metal1 maximum density (80%)", DRCSeverity.WARNING),
        ]
        if self.process_nm == 28:
            return rules
        # Scale for other processes
        scale = self.process_nm / 28.0
        return [DRCRule(r.category, r.layer, r.value * scale,
                        f"{r.description} ({self.process_nm}nm)",
                        r.severity) for r in rules]

    def check_width(self, shapes: List[Rect]) -> List[DRCViolation]:
        """Check minimum width rules."""
        viols = []
        width_rules = {r.layer: r for r in self.rules if r.category == DRCCategory.WIDTH}
        for shape in shapes:
            if shape.layer in width_rules:
                rule = width_rules[shape.layer]
                actual = min(shape.w, shape.h)
                if actual < rule.value:
                    viols.append(DRCViolation(rule, shape.x, shape.y, actual, rule.value, shape.name))
        return viols

    def check_spacing(self, shapes: List[Rect]) -> List[DRCViolation]:
        """Check minimum spacing rules."""
        viols = []
        space_rules = {}
        for r in self.rules:
            if r.category == DRCCategory.SPACING:
                space_rules.setdefault(r.layer, r)
        # Group by layer
        by_layer: Dict[int, List[Rect]] = {}
        for s in shapes:
            by_layer.setdefault(s.layer, []).append(s)
        for layer, layer_shapes in by_layer.items():
            if layer not in space_rules:
                continue
            rule = space_rules[layer]
            for i in range(len(layer_shapes)):
                for j in range(i + 1, len(layer_shapes)):
                    spacing = layer_shapes[i].spacing_to(layer_shapes[j])
                    if spacing < rule.value:
                        viols.append(DRCViolation(rule, layer_shapes[i].x, layer_shapes[i].y,
                                                  spacing, rule.value,
                                                  f"{layer_shapes[i].name}-{layer_shapes[j].name}"))
        return viols

    def check_enclosure(self, inner: List[Rect], outer: List[Rect],
                        outer_layer: int) -> List[DRCViolation]:
        """Check minimum enclosure rules."""
        viols = []
        enc_rules = [r for r in self.rules
                     if r.category == DRCCategory.ENCLOSURE and r.layer == outer_layer]
        if not enc_rules:
            return viols
        rule = enc_rules[0]
        for o_shape in outer:
            if o_shape.layer != outer_layer:
                continue
            for i_shape in inner:
                if not o_shape.contains(i_shape):
                    continue
                enc = o_shape.enclosure(i_shape)
                min_enc = min(enc[0], enc[1])
                if min_enc < rule.value:
                    viols.append(DRCViolation(rule, o_shape.x, o_shape.y,
                                              min_enc, rule.value, o_shape.name))
        return viols

    def check_density(self, shapes: List[Rect], die_area: float) -> List[DRCViolation]:
        """Check metal density rules."""
        viols = []
        density_rules = [r for r in self.rules if r.category == DRCCategory.DENSITY]
        by_layer: Dict[int, List[Rect]] = {}
        for s in shapes:
            by_layer.setdefault(s.layer, []).append(s)
        for layer, layer_shapes in by_layer.items():
            total_area = sum(s.area for s in layer_shapes)
            density = total_area / die_area if die_area > 0 else 0
            for rule in density_rules:
                if rule.layer != layer:
                    continue
                if "minimum" in rule.description and density < rule.value:
                    viols.append(DRCViolation(rule, 0, 0, density, rule.value,
                                              f"Layer {layer}"))
                elif "maximum" in rule.description and density > rule.value:
                    viols.append(DRCViolation(rule, 0, 0, density, rule.value,
                                              f"Layer {layer}"))
        return viols

    def run_all(self, shapes: List[Rect], die_area: float = 0) -> Dict:
        """Run all DRC checks."""
        self.violations = []
        self.violations.extend(self.check_width(shapes))
        self.violations.extend(self.check_spacing(shapes))
        self.violations.extend(self.check_enclosure(shapes, shapes, 2))  # M1 over contact
        if die_area > 0:
            self.violations.extend(self.check_density(shapes, die_area))

        errors = sum(1 for v in self.violations if v.rule.severity == DRCSeverity.ERROR)
        warnings = sum(1 for v in self.violations if v.rule.severity == DRCSeverity.WARNING)

        return {"total": len(self.violations), "errors": errors, "warnings": warnings,
                "clean": errors == 0, "violations": self.violations}


def demo():
    print("=== Design Rule Checker (28nm LP) ===\n")

    drc = DesignRuleChecker(process_nm=28)
    print(f"Process: {drc.process_nm}nm, Rules: {len(drc.rules)}")
    print()

    # Good shapes
    shapes = [
        Rect(0, 0, 10, 10, 2, "m1_wire_a"),       # Metal1 10um wide
        Rect(20, 0, 10, 10, 2, "m1_wire_b"),      # 10um spacing
        Rect(5, 5, 2, 2, 5, "contact"),            # Contact inside m1
        Rect(100, 100, 50, 50, 9, "pad_0"),        # Bonding pad
        Rect(200, 100, 50, 50, 9, "pad_1"),        # Another pad
        Rect(0, 50, 5, 5, 6, "active"),            # Active region
    ]

    result = drc.run_all(shapes, die_area=1000 * 1000)
    print("--- Clean Layout Check ---")
    print(f"  Total: {result['total']}, Errors: {result['errors']}, Warnings: {result['warnings']}")
    print(f"  Clean: {result['clean']}")
    print()

    # Add violations
    shapes_bad = shapes + [
        Rect(50, 50, 0.05, 0.05, 2, "m1_too_thin"),   # Width violation
        Rect(20.01, 0, 5, 5, 2, "m1_too_close"),        # Spacing violation
        Rect(150, 100, 10, 10, 9, "pad_too_small"),      # Pad too small
    ]

    result_bad = drc.run_all(shapes_bad, die_area=1000 * 1000)
    print("--- Layout With Violations ---")
    print(f"  Total: {result_bad['total']}, Errors: {result_bad['errors']}, Warnings: {result_bad['warnings']}")
    print(f"  Clean: {result_bad['clean']}")
    if result_bad['violations']:
        print("  Violations:")
        for v in result_bad['violations'][:10]:
            sev = "E" if v.rule.severity == DRCSeverity.ERROR else "W"
            print(f"    [{sev}] {v.shape_name}: {v.rule.description} "
                  f"(have {v.actual:.3f}um, need {v.expected:.3f}um)")

    print()
    # Process comparison
    print("--- Process Scaling ---")
    for nm in [28, 40, 65, 130]:
        d = DesignRuleChecker(nm)
        m1_rules = [r for r in d.rules if r.layer == 2 and r.category == DRCCategory.WIDTH]
        if m1_rules:
            print(f"  {nm:3d}nm Metal1 min width: {m1_rules[0].value:.4f}um")


if __name__ == "__main__":
    demo()
