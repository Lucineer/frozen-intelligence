# fi-manufacturing

---

# Manufacturing Playbook for Mask-Locked Inference Chips: From Tapeout to Mass Production

## Executive Summary
Mask-locked inference chips represent a paradigm shift in AI hardware security, where neural network weights are permanently encoded into the physical metallization layers during manufacturing. This playbook provides a comprehensive framework for scaling production from first silicon to high-volume manufacturing, balancing technical constraints, supply chain resilience, and cost optimization across multiple vessel classes.

---

## 1. First Tapeout Strategy: MPW for Proof of Concept

### 1.1 MPW Selection Criteria
- **Process Node**: SkyWater 130nm CMOS as baseline for initial validation
- **Shuttle Schedule**: Align with quarterly free shuttle programs for cost optimization
- **Die Size Allocation**: Target 2mm × 2mm for initial test vehicle
- **Test Structures**: Include 40% area for process monitoring and characterization

### 1.2 Initial Validation Objectives
- **Weight Encoding Verification**: Validate mask-level weight permanence through SEM/TEM cross-section analysis
- **Systolic Array Functionality**: Test 8×8 systolic array with fixed-weight MAC operations
- **Leakage Characterization**: Measure static power consumption of locked weights
- **Security Validation**: Attempt weight extraction through side-channel attacks (simulated)

### 1.3 Risk Mitigation
- **Redundant Design**: Include both mask-locked and SRAM-based weight storage for comparison
- **Process Corner Coverage**: Design for SS, TT, FF corners with ±20% voltage variation
- **Diagnostic Infrastructure**: Implement scan chains and boundary scan for debug

---

## 2. SkyWater 130nm Free Shuttle: $300 Validation Strategy

### 2.1 Resource Allocation
- **Area Budget**: 10mm² maximum per project
- **Test Vehicle Composition**:
  - 4mm²: 16×16 systolic array with mask-locked weights
  - 2mm²: Control logic and interface (AXI-Lite)
  - 2mm²: SRAM-based reference design
  - 2mm²: Process monitoring and test structures

### 2.2 Validation Priorities
1. **Weight Integrity**: Verify weight retention through 1,000 cycles of temperature stress (-40°C to 125°C)
2. **Functional Yield**: Target >80% functional die from shuttle run
3. **Performance Baseline**: Measure 8-bit MAC operations at 100MHz target frequency
4. **Power Profile**: Characterize dynamic/static power at various activation sparsity levels

### 2.3 Measurement Protocol
- **Parametric Testing**: 100% wafer probe with 5-point contact measurement
- **Structural Testing**: 95% fault coverage using combinational ATPG
- **Functional Testing**: Vector-based testing with 1,000 inference patterns
- **Reliability Screening**: 48-hour burn-in at 125°C for sample population

---

## 3. Production Foundry Selection Matrix

### 3.1 Evaluation Criteria
| Metric | Weight | TSMC | SMIC | GlobalFoundries |
|--------|--------|------|------|-----------------|
| **Security** | 25% | 9/10 (US-aligned) | 4/10 (geopolitical risk) | 8/10 (US-based) |
| **Process Maturity** | 20% | 10/10 (90nm-28nm) | 7/10 (130nm-40nm) | 9/10 (130nm-12nm) |
| **Mask Cost** | 15% | 5/10 ($500K-1M) | 8/10 ($200K-500K) | 7/10 ($300K-700K) |
| **Volume Capacity** | 15% | 10/10 (>100K wpm) | 9/10 (>50K wpm) | 8/10 (>30K wpm) |
| **IP Ecosystem** | 10% | 10/10 (complete) | 6/10 (limited) | 9/10 (comprehensive) |
| **Geopolitical Risk** | 15% | 8/10 (Taiwan risk) | 5/10 (US sanctions) | 10/10 (US/EU) |

### 3.2 Vessel-Specific Recommendations

**Scout/Messenger Class** (Low Power, <1 TOPS):
- **Primary**: GlobalFoundries 130nm LP
- **Backup**: SMIC 130nm LL
- **Rationale**: Cost-sensitive, mature process, US/EU supply chain preference

**Navigator Class** (Mid-Range, 1-10 TOPS):
- **Primary**: TSMC 90nm GP
- **Backup**: GlobalFoundries 90nm
- **Rationale**: Balance of performance, cost, and security

**Captain Class** (High Performance, 10-100 TOPS):
- **Primary**: TSMC 40nm LP
- **Backup**: GlobalFoundries 40nm
- **Rationale**: Performance-driven, higher integration density

### 3.3 Multi-Foundry Strategy
- **Phase 1** (0-10K units): Single foundry (GF 130nm)
- **Phase 2** (10K-100K units): Dual-source (GF + TSMC)
- **Phase 3** (>100K units): Triple-source with regional allocation

---

## 4. Yield Optimization Framework

### 4.1 Defect Density Modeling
```
Yield = [(1 + (A × D0)/α)^(-α)] × Y0
Where:
A = Die area (mm²)
D0 = Defect density (defects/cm²)
α = Clustering parameter (typically 0.5-2.0)
Y0 = Systematic yield component
```

### 4.2 Die Size Optimization
| Vessel Class | Target Area | Expected Yield @ 0.1 def/cm² | Cost/DIE |
|-------------|-------------|-----------------------------|----------|
| **Scout** | 16mm² | 92% | $1.20 |
| **Messenger** | 36mm² | 85% | $2.70 |
| **Navigator** | 64mm² | 78% | $5.10 |
| **Captain** | 144mm² | 65% | $12.50 |

### 4.3 Yield Enhancement Techniques
- **Redundant Vias**: 200% via duplication in weight metal layers
- **Wire Spreading**: 2× minimum spacing in critical weight routing
- **Dummy Fill Optimization**: <5% density variation across die
- **Guard Rings**: Triple-well isolation for analog/RF sections

### 4.4 Process-Specific Adjustments
- **TSMC**: Utilize CMP dummy fill recommendations
- **GF**: Implement recommended OPC rules for 130nm
- **SMIC**: Apply additional metal slotting rules

---

## 5. Test Strategy Architecture

### 5.1 Built-In Self-Test (BIST) Implementation
```
BIST Architecture:
- Weight Checksum: 32-bit CRC for mask-locked weight verification
- MAC Array BIST: March-C algorithm for multiplier testing
- Memory BIST: 8N algorithm for SRAM/register files
- Analog BIST: On-chip ADC/DAC for mixed-signal validation
```

### 5.2 Test Flow
1. **Wafer Sort** (100% coverage):
   - Continuity tests (opens/shorts)
   - IDDQ measurement (<10μA typical)
   - Digital functional test (95% coverage)
   - Weight checksum verification

2. **Package Test** (100% units):
   - Full temperature range (-40°C to 125°C)
   - Speed binning (3 bins: commercial, industrial, military)
   - Power consumption validation
   - Interface testing (PCIe, DDR, etc.)

3. **System-Level Test** (Sample basis):
   - Inference accuracy verification
   - Thermal performance
   - Long-term reliability

### 5.3 ATE Configuration
- **Platform**: Advantest V93K or Teradyne UltraFlex
- **Test Time Target**: <30 seconds per die
- **Parallel Testing**: 32 sites for Scout, 16 sites for Captain
- **Contact Resistance**: <1Ω per pin

### 5.4 Probe Card Strategy
- **Scout/Messenger**: MEMS vertical probe, 50μm pitch
- **Navigator/Captain**: Cantilever probe, 100μm pitch
- **Maintenance Schedule**: Clean every 50K touchdowns, replace tips every 500K

---

## 6. Packaging Technology Roadmap

### 6.1 QFN for Scout/Messenger
- **Size**: 5×5mm to 7×7mm
- **Lead Count**: 32 to 64 leads
- **Thermal**: Exposed pad, θJA = 35°C/W
- **Cost**: $0.15-$0.25 per unit at 100K volume
- **Assembly**: Strip-level molding, singulation after molding

### 6.2 BGA for Navigator/Captain
- **Size**: 10×10mm to 15×15mm
- **Ball Count**: 196 to 324 balls
- **Pitch**: 0.8mm for Navigator, 0.65mm for Captain
- **Substrate**: 2-2-2 laminate (4 layers total)
- **Cost**: $0.80-$1.50 per unit at 100K volume
- **Underfill**: Capillary flow for reliability enhancement

### 6.3 Advanced Packaging (Future)
- **2.5D Integration**: Silicon interposer for memory stacking
- **Fan-Out Wafer-Level**: For ultra-thin Scout variants
- **Thermal Solutions**: Integrated heat spreader for Captain class

### 6.4 Assembly Process Flow
```
1. Wafer Backgrind: 200μm final thickness
2. Wafer Dicing: Laser grooving + mechanical saw
3. Die Attach: Epoxy for QFN, solder for BGA
4. Wire Bond: 25μm Au wire, 2-3 mil loop height
5. Molding: Low-stress epoxy, <1% void content
6. Marking: Laser direct marking, 2D barcode
7. Singulation: Punch for QFN, saw for BGA array
8. Test: Final test, vision inspection
```

---

## 7. Supply Chain Architecture

### 7.1 Lead Time Analysis
| Component | Standard LT | Expedited LT | Buffer Stock |
|-----------|-------------|--------------|--------------|
| **Masks** | 8 weeks | 4 weeks | 2 sets |
| **Raw Wafers** | 12 weeks | 6 weeks | 1 month |
| **Packaging** | 6 weeks | 3 weeks | 2 weeks |
| **Test Sockets** | 10 weeks | 5 weeks | 10% spare |
| **Final Test** | 4 weeks | 2 weeks | N/A |

### 7.2 Dual Sourcing Strategy
- **Foundry**: Primary (GF) + Secondary (TSMC) with qualified masks
- **Packaging**: 2 assembly houses with identical process flows
- **Test**: Cross-qualified ATE programs between facilities
- **Materials**: Approved vendor list with 2+ suppliers per material

### 7.3 Inventory Management
```
Safety Stock = Z × √(LT × σD² + D² × σLT²)
Where:
Z = Service factor (1.65 for 95% service level)
LT = Lead time
σD = Demand standard deviation
D = Average demand
σLT = Lead time standard deviation
```

### 7.4 Risk Mitigation
- **Geographic Diversity**: US, Taiwan, and European supply chains
- **Buffer Capacity**: Maintain 20% excess capacity at each node
- **Vertical Integration**: Consider captive test for critical products
- **Long-term Agreements**: 12-24 month contracts with key suppliers

---

## 8. Quality Assurance Framework

### 8.1 Burn-in Conditions
| Stress Level | Temperature | Voltage | Duration | Application |
|-------------|-------------|---------|----------|-------------|
| **Standard** | 125°C | 1.1× Vdd | 48 hours | Commercial |
| **Extended** | 150°C | 1.2× Vdd | 168 hours | Industrial |
| **Accelerated** | 175°C | 1.3× Vdd | 1000 hours | Military/Aero |

### 8.2 Accelerated Life Testing
```
Arrhenius Model:
AF = exp[(Ea/k) × (1/Tuse - 1/Tstress)]
Where:
Ea = 0.7eV (typical for CMOS)
k = Boltzmann constant
T = Temperature in Kelvin

Test Conditions:
- High Temperature Operating Life: 1000 hours @ 150°C
- Temperature Cycling: 1000 cycles (-55°C to 150°C)
- Highly Accelerated Stress Test: 96 hours @ 130°C/85%RH
```

### 8.3 Lot Traceability
- **Wafer Level**: Fab lot ID, wafer number, coordinates
- **Assembly**: Date code, assembly line, operator
- **Test**: ATE serial, test program revision, results
- **Final**: 2D barcode with full genealogy

### 8.4 Failure Analysis Flow
1. **Electrical Characterization**: Identify failure signature
2. **Non-Destructive**: X-ray, CSAM, TDR
3. **Deprocessing**: Delayering to failure location
4. **Physical Analysis**: SEM, FIB, TEM
5. **Root Cause**: Process deviation, design marginality, material defect

---

## 9. Cost Optimization Engine

### 9.1 Cost Breakdown Analysis
```
Total Cost = (Wafer Cost/DPW) + Assembly + Test + Overhead

Where DPW (Die Per Wafer) = π × (R - S)² / A - π × (R - S) / √(2A)
R = Wafer radius
S = Edge exclusion
A = Die area
```

### 9.2 Design for Yield Initiatives
- **Area Reduction**: 10% shrink every 2 years through layout optimization
- **Test Time**: Target 20% reduction annually through DFT improvements
- **Package Cost**: Evaluate down-gauging opportunities quarterly
- **Material Cost**: Annual 5% reduction through supplier negotiations

### 9.3 Mature Process Advantages
- **130nm Benefits**:
  - Mask cost: $300K vs $1.5M for 28nm
  - Wafer cost: $800 vs $3,000 for 28nm
  - Yield learning: 6 months vs 18 months for new nodes
  - IP availability: Complete library with 10+ years of maturity
  - Reliability: FIT rate <10 vs 50-100 for advanced nodes

### 9.4 Die Shrinking Roadmap
- **Generation 1**: 130nm, baseline design
- **Generation 2**: 90nm, 40% area reduction
- **Generation 3**: 65nm, 60% area reduction from baseline
- **Generation 4**: 40nm, 75% area reduction from baseline

### 9.5 Volume-Based Cost Projections
| Annual Volume | Scout Cost | Navigator Cost | Captain Cost |
|---------------|------------|----------------|--------------|
| **100 units** | $45.00 | $120.00 | $280.00 |
| **1,000 units** | $18.50 | $52.00 | $135.00 |
| **10,000 units** | $8.20 | $24.50 | $68.00 |
| **100,000 units** | $4.10 | $13.80 | $38.50 |

---

## 10. Scaling Framework: 100 to 100K Units/Year

### 10.1 Phase-Based Scaling Strategy

**Phase 1: Prototype (100 units/year)**
- **Focus**: Design validation, customer sampling
- **Manufacturing**: MPW runs, manual assembly
- **Test**: Bench testing, limited automation
- **Team**: 5-10 engineers, hands-on everything
- **Key Metrics**: Functionality, performance, early reliability

**Phase 2: Low Volume (1,000 units/year)**
- **Focus**: Process qualification, yield learning
- **Manufacturing**: Dedicated mask set, semi-auto assembly
- **Test**: Basic ATE, 50% automation
- **Team**: 15-20 engineers, specialized roles emerging
- **Key Metrics**: Yield (>70%), test coverage (>90%), cost reduction

**Phase 3: Medium Volume (10,000 units/year)**
- **Focus**: Supply chain establishment, quality systems
- **Manufacturing**: Multi-project wafers, auto assembly
- **Test**: Full ATE, 80% automation, statistical process control
- **Team**: 30-40 engineers, dedicated manufacturing team
- **Key Metrics**: DPPM (<500), lead time (<8 weeks), dual sourcing

**Phase 4: High Volume (100,000 units/year)**
- **Focus**: Cost optimization, global distribution
- **Manufacturing**: Full wafers, multiple assembly sites
- **Test**: Parallel testing, 95% automation, predictive maintenance
- **Team**: 50+ engineers, regional support teams
- **Key Metrics**: Cost (<$5 for Scout), availability (>99%), scalability

### 10.2 Capacity Planning
```
Required Wafers/Month = (Units/Month) / (DPW × Yield)

Scaling Example (Scout Class):
- 100K units/year = 8,333 units/month
- DPW @ 130nm = 1,200 die/wafer
- Yield = 85%
- Required = 8,333 / (1,200 × 0.85) = 8.2 wafers/month
```

### 10.3 Infrastructure Investment Timeline
- **Year 1**: Test equipment, characterization lab
- **Year 2**: Assembly line setup, quality lab
- **Year 3**: Second source qualification, regional warehouses
- **Year 4**: Advanced packaging line, reliability lab
- **Year 5**: Full vertical integration capability

### 10.4 Risk Management at Scale
- **Technical**: Maintain 2-generation process roadmap
- **Supply**: 30% buffer capacity with key suppliers
- **Financial**: Hedge currency exposure