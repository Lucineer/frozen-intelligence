#!/usr/bin/env python3
"""Yield-aware swarm tiling for mask-locked inference chips.

Based on Chip Forge v2 concepts: MoE routing, defect tolerance,
zone-based die grading, and revenue optimization.
"""
import random, math, json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Zone(Enum):
    CORE = "core"
    MID = "mid"
    EDGE = "edge"

ZONE_COLORS = {Zone.CORE: "#22d3ee", Zone.MID: "#a78bfa", Zone.EDGE: "#fb923c"}
ZONE_DEFECT_MULT = {Zone.CORE: 0.3, Zone.MID: 0.7, Zone.EDGE: 1.5}
ZONE_QUALITY_THRESHOLD = {Zone.CORE: 0.38, Zone.MID: 0.68, Zone.EDGE: 0.92}


@dataclass
class ModelSpec:
    id: str
    name: str
    params: int
    zone: Zone
    category: str  # infrastructure, vision, audio, sensor, domain
    critical: bool = False
    role: str = "expert"  # router or expert


@dataclass
class Tile:
    x: int
    y: int
    zone: Zone
    dist: float
    model: Optional[str] = None
    defective: bool = False
    enabled: bool = True

    @property
    def id(self) -> str:
        return f"{self.x}_{self.y}"


class DieGrade(Enum):
    GOLD = ("GOLD", "#fbbf24", "Full swarm — flagship product", True)
    SILVER = ("SILVER", "#c8cdd8", "Core + most experts operational", True)
    BRONZE = ("BRONZE", "#fb923c", "Core functions only — budget SKU", True)
    SCRAP = ("SCRAP", "#ef4444", "Non-functional", False)

    def __init__(self, label, color, desc, sellable):
        self.label = label
        self.color = color
        self.desc = desc
        self.sellable = sellable


class DieGrid:
    """Generate and manage the physical die tile grid."""

    def __init__(self, cols: int = 8, rows: int = 8):
        self.cols = cols
        self.rows = rows
        self.tiles = self._generate()

    def _generate(self) -> List[Tile]:
        tiles = []
        cx = (self.cols - 1) / 2
        cy = (self.rows - 1) / 2
        max_dist = math.sqrt(cx * cx + cy * cy)
        for y in range(self.rows):
            for x in range(self.cols):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                norm = dist / max_dist if max_dist > 0 else 0
                if norm > 0.92:  # Circular mask
                    continue
                zone = Zone.CORE if norm < 0.38 else (Zone.MID if norm < 0.68 else Zone.EDGE)
                tiles.append(Tile(x, y, zone, norm))
        return tiles

    def simulate_defects(self, defect_rate: float) -> List[Tile]:
        """Simulate manufacturing defects."""
        return [Tile(t.x, t.y, t.zone, t.dist, t.model,
                     defective=random.random() < defect_rate * ZONE_DEFECT_MULT[t.zone],
                     enabled=not (random.random() < defect_rate * ZONE_DEFECT_MULT[t.zone]))
                for t in self.tiles]

    def assign_models(self, models: List[ModelSpec]) -> List[Tile]:
        """Assign models to tiles based on zone preferences."""
        result = []
        zone_tiles = {z: [t for t in self.tiles if t.zone == z] for z in Zone}
        for model in models:
            # Place critical models in their preferred zone
            preferred = zone_tiles.get(model.zone, [])
            if preferred:
                tile = preferred.pop(0)
                tile.model = model.id
                result.append(tile)
            # Fallback to any available
            else:
                for z in Zone:
                    if zone_tiles[z]:
                        tile = zone_tiles[z].pop(0)
                        tile.model = model.id
                        result.append(tile)
                        break
        # Add unassigned tiles
        for z in Zone:
            result.extend(zone_tiles[z])
        return result

    def grade(self, tiles: List[Tile]) -> DieGrade:
        """Grade a die based on surviving tiles."""
        total = len(tiles)
        alive = sum(1 for t in tiles if not t.defective)
        core_alive = sum(1 for t in tiles if t.zone == Zone.CORE and not t.defective)
        core_total = sum(1 for t in tiles if t.zone == Zone.CORE)
        ratio = alive / total if total > 0 else 0
        core_ratio = core_alive / core_total if core_total > 0 else 1

        if core_ratio < 0.8:
            return DieGrade.SCRAP
        if ratio >= 0.92 and core_ratio == 1:
            return DieGrade.GOLD
        if ratio >= 0.75:
            return DieGrade.SILVER
        if ratio >= 0.55:
            return DieGrade.BRONZE
        return DieGrade.SCRAP


class YieldSimulator:
    """Monte Carlo yield simulation."""

    @staticmethod
    def simulate(die: DieGrid, models: List[ModelSpec],
                 defect_rate: float, runs: int = 1000) -> Dict:
        results = {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "SCRAP": 0}
        expert_survival: Dict[str, Dict] = {}
        assigned = die.assign_models(models)

        for _ in range(runs):
            sim_tiles = die.simulate_defects(defect_rate)
            grade = die.grade(sim_tiles)
            results[grade.label] += 1

            for t in sim_tiles:
                if t.model:
                    if t.model not in expert_survival:
                        expert_survival[t.model] = {"alive": 0, "total": 0}
                    expert_survival[t.model]["total"] += 1
                    if not t.defective:
                        expert_survival[t.model]["alive"] += 1

        # Convert to percentages
        for k in results:
            results[k] = round(results[k] / runs * 100, 1)

        for k in expert_survival:
            es = expert_survival[k]
            es["rate"] = round(es["alive"] / es["total"] * 100, 1) if es["total"] > 0 else 0

        # Revenue estimate
        revenue = results["GOLD"] * 100 + results["SILVER"] * 60 + results["BRONZE"] * 25
        return {"distribution": results, "expert_survival": expert_survival,
                "revenue_index": round(revenue / 100, 1)}


class MoERouter:
    """Mixture-of-Experts routing for yield-aware chips."""

    def __init__(self, models: List[ModelSpec]):
        self.models = {m.id: m for m in models}
        self.router_model = next((m for m in models if m.role == "router"), None)
        self.experts = {m.id: m for m in models if m.role == "expert"}

    def route(self, input_type: str, available_experts: List[str]) -> Optional[str]:
        """Route input to best available expert."""
        if not available_experts:
            return None
        # Simple routing: match category to input type
        category_map = {"image": "vision", "audio": "audio", "sensor": "sensor",
                       "sonar": "domain", "gps": "domain", "all": "infrastructure"}
        category = category_map.get(input_type, "domain")
        # Priority: matching category → critical → any
        for expert_id in available_experts:
            m = self.models.get(expert_id)
            if m and m.category == category:
                return expert_id
        for expert_id in available_experts:
            m = self.models.get(expert_id)
            if m and m.critical:
                return expert_id
        return available_experts[0]

    def get_available(self, tiles: List[Tile]) -> List[str]:
        """Get list of alive expert tiles."""
        return [t.model for t in tiles if t.model and t.enabled and not t.defective]


class VerilogGenerator:
    """Generate Verilog for swarm chip top module."""

    def __init__(self, chip_name: str, models: List[ModelSpec]):
        self.chip_name = chip_name
        self.models = models
        self.router = next((m for m in models if m.role == "router"), None)
        self.experts = [m for m in models if m.role == "expert"]

    def generate_top(self, tiles: List[Tile]) -> str:
        """Generate top-level Verilog module."""
        lines = []
        n = self.chip_name
        ne = len(self.experts)
        nt = sum(1 for t in tiles if t.model)

        lines.append(f"// {n} — Mask-Locked Swarm Inference Chip")
        lines.append(f"// Generated by Lucineer Chip Forge v2")
        lines.append(f"// Architecture: Yield-Aware Mixture-of-Experts")
        lines.append(f"// Tiles: {nt}, Experts: {ne}, Zones: CORE/MID/EDGE")
        lines.append(f"")
        lines.append(f"module {n} #( ")
        lines.append(f"  parameter DATA_WIDTH  = 16,")
        lines.append(f"  parameter FRAC_BITS   = 8,")
        lines.append(f"  parameter NUM_EXPERTS = {ne},")
        lines.append(f"  parameter TILE_COUNT  = {nt}")
        lines.append(f") (")
        lines.append(f"  input  wire        clk,")
        lines.append(f"  input  wire        rst_n,")
        lines.append(f"  input  wire [TILE_COUNT-1:0] tile_enable,")
        lines.append(f"  input  wire                     valid_in,")
        lines.append(f"  input  wire [DATA_WIDTH-1:0]    data_in [0:127],")
        lines.append(f"  input  wire [7:0]               input_type,")
        lines.append(f"  output wire                     valid_out,")
        lines.append(f"  output wire [DATA_WIDTH-1:0]    data_out [0:31],")
        lines.append(f"  output wire [7:0]               expert_id")
        lines.append(f");")
        lines.append(f"")
        lines.append(f"  // MoE Router — always in CORE zone")
        lines.append(f"  wire [7:0] router_sel;")
        lines.append(f"  wire [NUM_EXPERTS-1:0] expert_avail;")

        for i, exp in enumerate(self.experts):
            lines.append(f"  wire expert_{i}_valid;")
            lines.append(f"  assign expert_avail[{i}] = tile_enable[{i}];")

        lines.append(f"")
        lines.append(f"  moe_router #(.DATA_WIDTH(DATA_WIDTH), .NUM_EXPERTS({ne})) u_router (")
        lines.append(f"    .clk(clk), .rst_n(rst_n),")
        lines.append(f"    .data_in(data_in), .input_type(input_type),")
        lines.append(f"    .expert_available(expert_avail), .selection(router_sel)")
        lines.append(f"  );")

        for i, exp in enumerate(self.experts):
            lines.append(f"")
            lines.append(f"  // Expert {i}: {exp.name} ({exp.params} params, {exp.zone.value} zone)")
            lines.append(f"  expert_tile #(.DATA_WIDTH(DATA_WIDTH), .EXPERT_ID({i})) u_expert_{i} (")
            lines.append(f"    .clk(clk), .rst_n(rst_n),")
            lines.append(f"    .enable(tile_enable[{i}]),")
            lines.append(f"    .selected(router_sel == {i}),")
            lines.append(f"    .valid_in(valid_in && router_sel == {i}),")
            lines.append(f"    .data_in(data_in),")
            lines.append(f"    .valid_out(expert_{i}_valid),")
            lines.append(f"    .data_out(/* ... */)")
            lines.append(f"  );")

        lines.append(f"")
        lines.append(f"endmodule")
        return "\n".join(lines)

    def generate_driver_header(self, tiles: List[Tile]) -> str:
        """Generate C driver header."""
        n = self.chip_name.upper()
        lines = []
        lines.append(f"#ifndef {n}_DRIVER_H")
        lines.append(f"#define {n}_DRIVER_H")
        lines.append(f"")
        lines.append(f"typedef struct {{")
        lines.append(f"  uint32_t tile_enable;   /* bitmask: 1=alive, 0=dead */")
        lines.append(f"  uint8_t  bin_grade;     /* 0=SCRAP, 1=BRONZE, 2=SILVER, 3=GOLD */")
        lines.append(f"  uint8_t  num_experts;")
        lines.append(f"}} {self.chip_name}_config_t;")
        lines.append(f"")

        for i, exp in enumerate(self.experts):
            lines.append(f"#define EXPERT_{exp.id.upper()} {i}  /* {exp.name} — {exp.zone.value} */")

        lines.append(f"")
        lines.append(f"#define INPUT_IMAGE   0x01")
        lines.append(f"#define INPUT_AUDIO   0x02")
        lines.append(f"#define INPUT_SENSOR  0x03")
        lines.append(f"")
        lines.append(f"int {self.chip_name}_infer({self.chip_name}_config_t *cfg,")
        lines.append(f"    uint8_t input_type, const int16_t *data_in,")
        lines.append(f"    int16_t *data_out, uint8_t *expert_used);")
        lines.append(f"")
        lines.append(f"#endif")
        return "\n".join(lines)


# Standard model library
DECKBOSS_MODELS = [
    ModelSpec("router", "MoE Router", 15000, Zone.CORE, "infrastructure", True, "router"),
    ModelSpec("safety", "Safety Watchdog", 8000, Zone.CORE, "infrastructure", True),
    ModelSpec("primary_cls", "Primary Classifier", 200000, Zone.CORE, "vision", True),
    ModelSpec("feat_ext_a", "Feature Extractor A", 150000, Zone.MID, "vision"),
    ModelSpec("feat_ext_b", "Feature Extractor B", 150000, Zone.MID, "vision"),
    ModelSpec("audio_cmd", "Audio Command", 80000, Zone.MID, "audio"),
    ModelSpec("sensor_fuse", "Sensor Fusion", 100000, Zone.MID, "sensor"),
    ModelSpec("anomaly_a", "Vibration Anomaly", 50000, Zone.MID, "sensor"),
    ModelSpec("obj_detect", "Object Detector", 300000, Zone.MID, "vision"),
    ModelSpec("depth_est", "Depth Estimator", 120000, Zone.MID, "sensor"),
    ModelSpec("species_sonar", "Species (Sonar)", 180000, Zone.EDGE, "domain"),
    ModelSpec("species_visual", "Species (Visual)", 250000, Zone.EDGE, "domain"),
    ModelSpec("weather", "Weather Pattern", 60000, Zone.EDGE, "domain"),
    ModelSpec("ice_detect", "Ice Detection", 90000, Zone.EDGE, "domain"),
    ModelSpec("catch_weight", "Catch Estimator", 40000, Zone.EDGE, "domain"),
    ModelSpec("bycatch", "Bycatch Detector", 150000, Zone.EDGE, "domain"),
    ModelSpec("gear_status", "Gear Monitor", 30000, Zone.EDGE, "sensor"),
    ModelSpec("nav_predict", "Nav Predictor", 70000, Zone.EDGE, "domain"),
    ModelSpec("diagnostic", "Self-Diagnostic", 20000, Zone.EDGE, "infrastructure"),
    ModelSpec("experimental", "Experimental Slot", 100000, Zone.EDGE, "infrastructure"),
]


def demo():
    print("=== Yield-Aware Swarm Tiling ===\n")

    # Create die
    die = DieGrid(8, 8)
    print(f"Die: 8×8 grid = {len(die.tiles)} tiles")
    for zone in Zone:
        count = sum(1 for t in die.tiles if t.zone == zone)
        print(f"  {zone.value:5s}: {count} tiles")
    print()

    # Assign models
    assigned = die.assign_models(DECKBOSS_MODELS)
    modeled = sum(1 for t in assigned if t.model)
    print(f"Models assigned: {modeled}/{len(DECKBOSS_MODELS)}")
    total_params = sum(m.params for m in DECKBOSS_MODELS)
    print(f"Total parameters: {total_params:,} ({total_params/1000:.0f}K)")
    print()

    # Simulate defects
    print("--- Defect Simulation (5% defect rate) ---")
    sim = die.simulate_defects(0.05)
    dead = sum(1 for t in sim if t.defective)
    print(f"  Dead tiles: {dead}/{len(sim)}")
    grade = die.grade(sim)
    print(f"  Grade: {grade.label} — {grade.desc}")
    print()

    # Monte Carlo yield
    print("--- Monte Carlo Yield (1000 runs) ---")
    sim = YieldSimulator()
    for defect_rate in [0.03, 0.05, 0.08, 0.12]:
        result = sim.simulate(die, DECKBOSS_MODELS, defect_rate, 500)
        d = result["distribution"]
        print(f"  {defect_rate:.0%} defects: GOLD={d['GOLD']:5.1f}% SILVER={d['SILVER']:5.1f}% "
              f"BRONZE={d['BRONZE']:5.1f}% SCRAP={d['SCRAP']:5.1f}%  "
              f"Revenue: {result['revenue_index']:.1f}")
    print()

    # MoE routing
    print("--- MoE Router ---")
    router = MoERouter(DECKBOSS_MODELS)
    sim_tiles = die.simulate_defects(0.05)
    available = router.get_available(sim_tiles)
    print(f"  Available experts: {len(available)}/{len(DECKBOSS_MODELS)}")
    for inp in ["image", "audio", "sensor", "sonar"]:
        selected = router.route(inp, available)
        model = router.models.get(selected, None)
        print(f"  Input '{inp:7s}' → {model.name if model else selected}")
    print()

    # Verilog generation
    print("--- Verilog Generation ---")
    gen = VerilogGenerator("deckboss_marine_v1", DECKBOSS_MODELS)
    verilog = gen.generate_top(assigned)
    print(f"  Top module: {len(verilog)} chars, {verilog.count(chr(10))} lines")
    header = gen.generate_driver_header(assigned)
    print(f"  Driver header: {len(header)} chars")
    print(f"\n  Sample Verilog (first 10 lines):")
    for line in verilog.split("\n")[:10]:
        print(f"    {line}")


if __name__ == "__main__":
    demo()
