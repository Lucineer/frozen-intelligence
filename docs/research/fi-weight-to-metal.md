# fi-weight-to-metal

---

# **Weight-to-Metal Compilation Pipeline for Mask-Locked Inference Chips**

## **1. Architectural Foundation: The Metal-as-Weight Paradigm**

### **Core Concept**
In mask-locked inference chips, weights cease to be stored values and become **physical interconnect topology**. Each weight value maps to a specific metal layer configuration that creates permanent computational pathways. This transforms the chip from a programmable device into a **physical instantiation of the neural network**.

### **Key Innovations Required:**
- **Weight-to-Geometry Mapping**: Mathematical mapping from quantized values to physical dimensions
- **Metal Layer Encoding**: Using metal width, spacing, and layer assignment as weight representation
- **Fixed-Function Systolic Fabric**: Hardwired dataflow with weights embedded in routing
- **Mask-Locked Inference**: Single-purpose silicon with no weight reconfiguration capability

---

## **2. Complete Pipeline Architecture**

### **Pipeline Overview:**
```
PyTorch Model → Quantization → Geometry Mapping → METL Encoding → Layout Compilation → DRC → GDSII
```

---

## **3. Detailed Pipeline Components**

### **3.1 Input Format: PyTorch State_dict Processing**

**Custom Parser Architecture:**
```python
class WeightExtractor:
    def __init__(self):
        self.layer_graph = nx.DiGraph()
        self.weight_tensors = {}
        self.connection_map = {}
    
    def parse_state_dict(self, state_dict):
        # Extract computational graph from PyTorch model
        # Identify layer types and connectivity
        # Separate weights from biases and normalization parameters
        # Build directed acyclic graph of operations
        
    def categorize_weights(self):
        # Classify weights by layer type for mixed-precision assignment
        # Attention: Q, K, V, O projections
        # FFN: gate, up, down projections
        # Embeddings: token, position
        # Normalization: LayerNorm parameters
```

**Processing Steps:**
1. **Graph Extraction**: Convert PyTorch model to computational DAG
2. **Weight Clustering**: Group weights by precision requirements
3. **Spatial Analysis**: Determine systolic array placement requirements
4. **Memory Hierarchy Mapping**: Convert virtual memory access to physical dataflow

---

## **4. Quantization Pipeline: Mixed-Precision Assignment**

### **4.1 Precision Strategy**
```
Layer Type      | Precision | Rationale
----------------|-----------|-------------------------------------------
Embeddings      | INT8      | High dynamic range, semantic sensitivity
LayerNorm       | FP32      | Statistical normalization requires precision
Attention QKV   | INT4      | Dot-product tolerant to quantization
Attention O     | INT4      | Output projection
FFN Gate/Up     | INT4      | Gating mechanisms
FFN Down        | INT4      | Dimensionality reduction
Residual Adds   | FP16      | Accumulation precision
```

### **4.2 Custom Quantization Engine**
```python
class MetalAwareQuantizer:
    def __init__(self, metal_layers=8):
        self.metal_layers = metal_layers
        self.geometry_constraints = self.load_foundry_rules()
        
    def mixed_precision_quantize(self, weight_tensor, layer_type):
        # 1. Analyze weight distribution
        stats = self.compute_statistics(weight_tensor)
        
        # 2. Select precision based on layer type and sensitivity
        precision_config = self.get_precision_config(layer_type)
        
        # 3. Quantize with metal-aware rounding
        # Rounding considers physical realizability
        quantized = self.metal_aware_round(
            weight_tensor, 
            precision_config,
            self.geometry_constraints
        )
        
        # 4. Encode with error correction for manufacturing variations
        encoded = self.add_manufacturing_margin(quantized)
        
        return encoded, precision_config
    
    def metal_aware_round(self, values, precision, constraints):
        # Round to nearest physically realizable value
        # Physical realizability determined by:
        # - Minimum metal width (e.g., 40nm in 28nm process)
        # - Minimum spacing (e.g., 40nm)
        # - Discrete metal layer choices (M1-M8)
        
        # Create mapping table from quantized values to physical parameters
        physical_values = []
        for val in values:
            # Find closest physically realizable configuration
            phys_config = self.find_closest_physical_config(
                val, 
                precision,
                constraints
            )
            physical_values.append(phys_config)
        
        return physical_values
```

### **4.3 Calibration Methodology**
- **Statistical Calibration**: Per-layer sensitivity analysis
- **Manufacturing-Aware Calibration**: Account for process variations
- **Temperature Compensation**: Design for thermal stability
- **Aging Compensation**: Account for electromigration effects

---

## **5. Weight-to-Geometry Conversion**

### **5.1 Fundamental Mapping Principle**
Each weight value maps to a **3D metal interconnect configuration**:

```
Weight Value → [Metal Layer, Width, Length, Spacing, Via Pattern]
```

### **5.2 Encoding Scheme**
For INT4 weights (16 possible values):

```python
class WeightToGeometryMapper:
    # Mapping table: INT4 value → Physical configuration
    GEOMETRY_MAP = {
        0:  {'layer': 'M1', 'width': 0.04, 'spacing': 0.04, 'via': 'none'},
        1:  {'layer': 'M1', 'width': 0.05, 'spacing': 0.04, 'via': 'down'},
        2:  {'layer': 'M1', 'width': 0.06, 'spacing': 0.05, 'via': 'up'},
        # ... 13 more mappings
        15: {'layer': 'M3', 'width': 0.12, 'spacing': 0.08, 'via': 'both'}
    }
    
    # For INT8 (256 values), use combination encoding:
    # [Metal Layer Pair] × [Width] × [Spacing] × [Via Configuration]
    
    def map_weight_to_geometry(self, quantized_weight, layer_type):
        # 1. Get base geometry from mapping table
        base_config = self.GEOMETRY_MAP[quantized_weight]
        
        # 2. Apply layer-type specific adjustments
        adjusted = self.apply_layer_adjustments(base_config, layer_type)
        
        # 3. Ensure DRC compliance
        drc_checked = self.apply_drc_constraints(adjusted)
        
        # 4. Generate physical coordinates
        physical_layout = self.generate_polygons(drc_checked)
        
        return physical_layout
```

### **5.3 Multi-Layer Encoding for High Precision**
For FP32 LayerNorm parameters:
- Use **composite encoding** across multiple metal layers
- **8 metal layers** × **4 width variations** × **4 spacing variations** = 128 configurations
- Combine adjacent wires for higher precision: 128² = 16,384 values
- Add **capacitive coupling** effects for analog precision

### **5.4 Differential Encoding for Noise Immunity**
```python
def create_differential_pair(weight_value):
    # For each weight, create complementary positive/negative paths
    # Improves noise immunity and manufacturing tolerance
    
    pos_config = map_positive(weight_value)
    neg_config = map_negative(weight_value)
    
    # Ensure symmetric routing for common-mode rejection
    return symmetric_layout(pos_config, neg_config)
```

---

## **6. Binary Encoding: METL Format Specification**

### **6.1 METL (Metal Encoding Transfer Language) Format**
```
METL File Structure:
[Header]
VERSION: 2.0
FOUNDRY: SMIC_28
DESIGN: QWEN3.5-3B
TIMESTAMP: 2024-01-15T10:30:00Z

[Layer Definitions]
LAYER M1: METAL1 0.04um MIN_WIDTH 0.04um MIN_SPACE
LAYER M2: METAL2 0.04um MIN_WIDTH 0.04um MIN_SPACE
...
LAYER M8: METAL8 0.12um MIN_WIDTH 0.08um MIN_SPACE

[Cell Definitions]
CELL MAC_ARRAY_16x16:
  TYPE: SYSTOLIC
  DIMENSIONS: 16x16
  INPUT_PORTS: NORTH, WEST
  OUTPUT_PORTS: SOUTH, EAST
  
CELL LAYERNORM_UNIT:
  TYPE: NORMALIZATION
  PRECISION: FP32
  INPUTS: 512
  OUTPUTS: 512

[Placement]
INSTANCE ARRAY_0_0: MAC_ARRAY_16x16 AT (0, 0)
INSTANCE ARRAY_0_1: MAC_ARRAY_16x16 AT (200, 0)
...
INSTANCE LN_12: LAYERNORM_UNIT AT (3200, 1500)

[Routing]
# Weight encoding as metal interconnect
ROUTE WEIGHT_W0_0:
  LAYER: M3
  WIDTH: 0.06um
  COORDINATES: [(0,0), (10,0), (10,5), (15,5)]
  CONNECTS: ARRAY_0_0.PORT_W0 -> ARRAY_0_0.MAC_0_0
  
ROUTE WEIGHT_W1_0:
  LAYER: M4
  WIDTH: 0.08um
  COORDINATES: [(0,2), (12,2), (12,7), (15,7)]
  CONNECTS: ARRAY_0_0.PORT_W1 -> ARRAY_0_0.MAC_0_1

[Weight Encoding Table]
# Maps quantized values to METL configurations
ENCODING INT4:
  0: {LAYER: M1, WIDTH: 0.04, SPACE: 0.04}
  1: {LAYER: M2, WIDTH: 0.05, SPACE: 0.04}
  ...
  15: {LAYER: M3, WIDTH: 0.12, SPACE: 0.08}

[Verification]
DRC_RULES: SMIC_28_DENSE
LVS_NETLIST: EXTRACTED_FROM_METL
ERC_RULES: POWER_DENSITY_MAX_0.5mW/um2
```

### **6.2 METL Compiler Architecture**
```python
class METLCompiler:
    def __init__(self, foundry="smic_28"):
        self.foundry_rules = self.load_foundry_pdk(foundry)
        self.metal_layers = self.foundry_rules['metal_layers']
        
    def compile_to_metl(self, placed_layout, weight_mappings):
        # 1. Generate header with design metadata
        metl_content = self.generate_header()
        
        # 2. Define metal layers based on foundry PDK
        metl_content += self.define_layers()
        
        # 3. Instantiate systolic arrays and processing elements
        metl_content += self.instantiate_arrays(placed_layout)
        
        # 4. Encode weights as metal interconnect
        metl_content += self.encode_weights(weight_mappings)
        
        # 5. Route data paths between arrays
        metl_content += self.route_data_paths(placed_layout)
        
        # 6. Add power distribution network
        metl_content += self.add_power_grid()
        
        # 7. Include verification commands
        metl_content += self.add_verification()
        
        return metl_content
    
    def encode_weights(self, weight_mappings):
        metl_code = "\n[Weight Encoding]\n"
        
        for array_name, weights in weight_mappings.items():
            for i, weight in enumerate(weights):
                # Get physical configuration for this weight value
                config = self.weight_to_config(weight)
                
                # Generate METL routing command
                route_cmd = self.generate_route_command(
                    array_name, i, config
                )
                metl_code += route_cmd + "\n"
        
        return metl_code
```

---

## **7. Layout Compiler: Systolic Array Placement**

### **7.1 Hierarchical Design Strategy**
```
Top Level: Chip
├── Memory Hierarchy
│   ├── Input Buffer Banks
│   ├── Intermediate Activation FIFOs
│   └── Output Buffer Banks
├── Compute Fabric
│   ├── Attention Engine (Nx systolic arrays)
│   ├── FFN Engine (Mx systolic arrays)
│   └── Normalization Units
├── Control Fabric
│   ├── Finite State Machines
│   ├── Clock Distribution
│   └── Power Management
└── I/O Ring
    ├── High-Speed SerDes
    ├── Memory Interfaces
    └── Test Access Ports
```

### **7.2 Systolic Array Architecture**
**16x16 Weight-Stationary Systolic Array:**
- **Weight Encoding**: Metal interconnect within each MAC cell
- **Data Flow**: Activation streaming from north/west, results to south/east
- **Control**: Minimal FSM for timing coordination

**Placement Algorithm:**
```python
class SystolicPlacer:
    def place_arrays(self, model_graph, die_size):
        # 1. Extract computational requirements
        ops_per_layer = self.analyze_operations(model_graph)
        
        # 2. Determine array count per layer type
        attention_arrays = self.calc_arrays_needed(
            ops_per_layer['attention'], 
            array_size=16
        )
        
        ffn_arrays = self.calc_arrays_needed(
            ops_per_layer['ffn'], 
            array_size=16
        )
        
        # 3. Floorplan with dataflow optimization
        placement = self.dataflow_aware_floorplan(
            attention_arrays,
            ffn_arrays,
            die_size
        )
        
        # 4. Insert routing channels
        placement = self.add_routing_channels(placement)
        
        # 5. Optimize for wirelength and timing
        placement = self.optimize_placement(placement)
        
        return placement
    
    def dataflow_aware_floorplan(self, attention_arrays, ffn_arrays, die_size):
        # Place attention arrays in grid
        # Place FFN arrays adjacent with minimal routing
        # Insert shared normalization units
        # Ensure balanced wirelength and timing
        
        # Use simulated annealing with custom cost function:
        # Cost = α × Wirelength + β × TimingCriticality + γ × PowerDensity
        
        return optimized_placement
```

### **7.3 Routing Architecture**
**Three-Level Routing Hierarchy:**
1. **Local Routing**: Within systolic arrays (metal layers M1-M3)
2. **Intermediate Routing**: Between arrays (M4-M6)
3. **Global Routing**: Chip-level data movement (M7-M8)

**Custom Router Implementation:**
```python
class MetalWeightRouter:
    def route_weight_interconnect(self, weight_values, target_array):
        # Convert weight values to physical metal patterns
        # Route from weight input ports to individual MAC cells
        
        routes = []
        for i, weight in enumerate(weight_values):
            # Get geometry for this weight value
            geometry = self.weight_to_geometry(weight)
            
            # Create metal shape
            metal_shape = self.create_metal_shape(
                start_point=weight_input_port(i),
                end_point=target_array.mac_cell(i),
                geometry_config=geometry
            )
            
            routes.append(metal_shape)
        
        return routes
    
    def create_metal_shape(self, start, end, config):
        # Generate Manhattan routing with layer changes
        # Respect minimum width/spacing rules
        # Add vias for layer transitions
        
        path = self.manhattan_route(start, end)
        
        # Convert path to physical metal polygons
        polygons = []
        current_layer = config['layer']
        
        for segment in path:
            # Create rectangle for each wire segment
            rect = self.create_wire_rectangle(
                segment, 
                config['width'],
                current_layer
            )
            polygons.append(rect)
            
            # Add via if layer changes
            if segment.layer_change:
                via = self.create_via(
                    segment.end,
                    from_layer=current_layer,
                    to_layer=segment.new_layer
                )
                polygons.append(via)
                current_layer = segment.new_layer
        
        return polygons
```

---

## **8. DRC (Design Rule Check) Engine**

### **8.1 Foundry-Specific Rule Deck**
**SMIC 28nm DRC Rules (Example Subset):**
```
# Metal Rules
M1.min_width = 0.04um
M1.min_space = 0.04um
M1.min_area = 0.05um²

M2.min_width = 0.04um  
M2.min_space = 0.04um
M2.min_area = 0.05um²

M8.min_width = 0.12um
M8.min_space = 0.08um
M8.min_area = 0.5um²

# Via Rules
VIA1.min_size = 0.05x0.05um
VIA1.min_enclosure = 0.01um

# Density Rules
METAL_DENSITY.min = 20%
METAL_DENSITY.max = 80%

# Antenna Rules
GATE_ANTENNA_RATIO.max = 400
```

### **8.2 Custom DRC for Weight Encoding**
```python
class WeightEncodingDRC:
    def check_weight_encoding(self, metal_layout, weight_mappings):
        violations = []
        
        # 1. Check weight-to-geometry mapping compliance
        for weight_id, geometry in weight_mappings.items():
            # Verify geometry is physically realizable
            if not self.is_physically_realizable(geometry):
                violations.append(f"Weight {weight_id}: Unrealizable geometry")
            
            # Check for manufacturing variations
            if not self.has_manufacturing_margin(geometry):
                violations.append(f"Weight {weight_id}: Insufficient margin")
        
        # 2. Check metal density for CMP (Chemical Mechanical Polishing)
        density_map = self.compute_metal_density(metal_layout)
        if not self.check_density_rules(density_map):
            violations.append("Metal density violation")
        
        # 3. Check electromigration limits
        current