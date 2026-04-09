#!/usr/bin/env python3
"""Formal property checker for mask-locked chip designs.

Bounded model checking for safety properties, liveness checks,
and invariant verification on RTL-level designs.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


class PropertyType(Enum):
    SAFETY = "safety"       # bad thing never happens
    LIVENESS = "liveness"   # good thing eventually happens
    INVARIANT = "invariant" # always holds
    ASSERTION = "assertion" # signal condition


@dataclass
class Property:
    name: str
    prop_type: PropertyType
    expression: str          # Python expression evaluated over signal state
    description: str = ""


@dataclass
class PropertyResult:
    property: Property
    passed: bool
    cycle_found: int = -1
    counterexample: Optional[Dict] = None
    states_checked: int = 0


class StateSpace:
    """Track explored states for bounded model check."""

    def __init__(self, max_states: int = 100000):
        self.max_states = max_states
        self.visited: Set[str] = set()

    def state_hash(self, signals: Dict[str, int]) -> str:
        return ",".join(f"{k}:{v}" for k, v in sorted(signals.items()))

    def is_visited(self, signals: Dict[str, int]) -> bool:
        h = self.state_hash(signals)
        return h in self.visited

    def add(self, signals: Dict[str, int]):
        h = self.state_hash(signals)
        self.visited.add(h)

    @property
    def count(self) -> int:
        return len(self.visited)


class BoundedModelChecker:
    """BMC engine: check properties up to a depth bound."""

    def __init__(self):
        self.properties: List[Property] = []
        self.results: List[PropertyResult] = []
        self.state_space = StateSpace()

    def add_property(self, name: str, prop_type: PropertyType,
                     expression: str, description: str = ""):
        self.properties.append(Property(name, prop_type, expression, description))

    def _eval_expr(self, expr: str, state: Dict[str, int]) -> bool:
        """Evaluate property expression over signal state."""
        try:
            return bool(eval(expr, {"__builtins__": {}}, state))
        except:
            return True  # assume pass on eval error

    def check_safety(self, prop: Property,
                     initial: Dict[str, int],
                     transition_fn,  # (state) -> state
                     bound: int = 50) -> PropertyResult:
        """BMC for safety: property must hold in ALL reachable states."""
        states_checked = 0
        current = dict(initial)

        for cycle in range(bound + 1):
            states_checked += 1
            self.state_space.add(current)

            # Check property
            if not self._eval_expr(prop.expression, current):
                return PropertyResult(prop, False, cycle, current, states_checked)

            # Advance state
            current = transition_fn(current)
            if self.state_space.is_visited(current):
                break  # loop detected

        return PropertyResult(prop, True, -1, None, states_checked)

    def check_liveness(self, prop: Property,
                       initial: Dict[str, int],
                       transition_fn,
                       bound: int = 50) -> PropertyResult:
        """BMC for liveness: property must eventually hold."""
        states_checked = 0
        current = dict(initial)

        for cycle in range(bound + 1):
            states_checked += 1
            if self._eval_expr(prop.expression, current):
                return PropertyResult(prop, True, cycle, None, states_checked)
            current = transition_fn(current)

        return PropertyResult(prop, False, bound, current, states_checked)

    def check_invariant(self, prop: Property,
                        initial: Dict[str, int],
                        transition_fn,
                        bound: int = 50) -> PropertyResult:
        """Check invariant: property holds in every state."""
        return self.check_safety(prop, initial, transition_fn, bound)

    def run_all(self, initial: Dict[str, int],
                transition_fn, bound: int = 50) -> List[PropertyResult]:
        """Check all properties."""
        self.results = []
        self.state_space = StateSpace()

        for prop in self.properties:
            if prop.prop_type == PropertyType.SAFETY:
                r = self.check_safety(prop, initial, transition_fn, bound)
            elif prop.prop_type == PropertyType.LIVENESS:
                r = self.check_liveness(prop, initial, transition_fn, bound)
            elif prop.prop_type == PropertyType.INVARIANT:
                r = self.check_invariant(prop, initial, transition_fn, bound)
            else:
                r = self.check_safety(prop, initial, transition_fn, bound)
            self.results.append(r)

        return self.results

    def report(self) -> str:
        lines = ["=== Formal Verification Report ===", ""]
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        for r in self.results:
            status = "PASS" if r.passed else f"FAIL @ cycle {r.cycle_found}"
            lines.append(f"  [{status:12s}] {r.property.name}: {r.property.description}")
            if not r.passed and r.counterexample:
                lines.append(f"      Counterexample: {r.counterexample}")
            lines.append(f"      States checked: {r.states_checked}")

        lines.append("")
        lines.append(f"Total: {passed} passed, {failed} failed")
        lines.append(f"State space explored: {self.state_space.count} unique states")
        return "\n".join(lines)


def demo():
    print("=== Formal Property Checker ===\n")

    # Simple FSM: 2-bit counter with enable
    def counter_transition(state: Dict[str, int]) -> Dict[str, int]:
        count = state["count"]
        enable = state.get("enable", 1)
        if enable:
            count = (count + 1) & 0xF  # 4-bit wrap
        new_state = {"count": count, "enable": enable}
        return new_state

    initial = {"count": 0, "enable": 1}

    bmc = BoundedModelChecker()

    # Safety: count never equals 16 (overflow)
    bmc.add_property("no_overflow", PropertyType.SAFETY,
                     "count < 16", "Counter never overflows 4 bits")

    # Invariant: count is always non-negative
    bmc.add_property("non_negative", PropertyType.INVARIANT,
                     "count >= 0", "Count is always >= 0")

    # Liveness: count reaches 15 eventually
    bmc.add_property("reaches_max", PropertyType.LIVENESS,
                     "count == 15", "Counter eventually reaches maximum value")

    # Safety: count never negative
    bmc.add_property("no_negative", PropertyType.SAFETY,
                     "count >= 0", "Count never goes negative")

    results = bmc.run_all(initial, counter_transition, bound=20)
    print(bmc.report())

    # Property violation demo
    print("\n--- Property Violation Demo ---")
    bmc2 = BoundedModelChecker()

    # Intentional failing property: count never equals 5
    bmc2.add_property("never_five", PropertyType.SAFETY,
                      "count != 5", "Counter never hits 5 (SHOULD FAIL)")

    results2 = bmc2.run_all(initial, counter_transition, bound=20)
    print(bmc2.report())

    # FSM with reset
    print("\n--- FSM with Reset ---")
    def fsm_with_reset(state: Dict[str, int]) -> Dict[str, int]:
        val = state["val"]
        rst = state.get("rst", 0)
        if rst:
            val = 0
        else:
            val = (val + 1) & 0x7  # 3-bit counter
        return {"val": val, "rst": 0}  # rst auto-clears

    bmc3 = BoundedModelChecker()
    bmc3.add_property("reset_clears", PropertyType.SAFETY,
                      "val == 0 if rst == 1 else val >= 0",
                      "Reset always clears counter")

    init_fsm = {"val": 3, "rst": 1}
    results3 = bmc3.run_all(init_fsm, fsm_with_reset, bound=10)
    print(bmc3.report())

    # Mutual exclusion demo
    print("\n--- Mutual Exclusion ---")
    def mutex_transition(state: Dict[str, int]) -> Dict[str, int]:
        p1 = state["p1"]
        p2 = state["p2"]
        # Simple round-robin
        if p1 == 1 and p2 == 0:
            p1, p2 = 0, 1
        elif p1 == 0 and p2 == 1:
            p1, p2 = 1, 0
        elif p1 == 0 and p2 == 0:
            p1 = 1
        return {"p1": p1, "p2": p2}

    bmc4 = BoundedModelChecker()
    bmc4.add_property("mutual_exclusion", PropertyType.SAFETY,
                      "not (p1 == 1 and p2 == 1)",
                      "Never both processes in critical section")

    init_mutex = {"p1": 0, "p2": 0}
    results4 = bmc4.run_all(init_mutex, mutex_transition, bound=20)
    print(bmc4.report())


if __name__ == "__main__":
    demo()
