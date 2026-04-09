#!/usr/bin/env python3
"""Frozen Intelligence CLI — unified toolchain for mask-locked chips.

Usage:
  python -m frozen_intelligence compile --model qwen3.5-3b --precision int4 --output chip.bin
  python -m frozen_intelligence verify --chip scout --clock 500 --power 1.0
  python -m frozen_intelligence estimate --model 3b --process 28nm --yield 85
  python -m frozen_intelligence simulate --model messenger --prompt "Hello" --tokens 64
  python -m frozen_intelligence benchmark --iterations 100
"""
import argparse, sys, json, time, math
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_compile(args):
    """Compile model weights to chip binary."""
    print(f"Compiling {args.model} with {args.precision} precision...")
    from weight_compiler import WeightCompiler
    compiler = WeightCompiler()
    weights = compiler.generate_model(args.model, hidden_dim=768, num_layers=24)
    chip = compiler.compile(args.model, weights)
    print(chip.summary())
    if args.output:
        with open(args.output, "wb") as f:
            f.write(chip.binary)
        print(f"Binary written to {args.output} ({len(chip.binary)} bytes)")
    return 0


def cmd_verify(args):
    """Verify chip meets timing/power constraints."""
    print(f"Verifying {args.chip} @ {args.clock}MHz, {args.power}W...")
    from chip_verifier import ChipVerifier
    verifier = ChipVerifier(process_nm=28, clock_mhz=args.clock)
    # Run timing checks
    checks = verifier.run_timing_checks()
    passes = sum(1 for c in checks if c.passed)
    print(f"Timing: {passes}/{len(checks)} passed")
    # Power check
    from clock_gating import PowerEstimator
    est = PowerEstimator()
    p = est.full_estimate(3.0 if args.chip == "messenger" else 1.0)
    print(f"Power estimate: {p['total_w']:.1f}W vs budget {args.power}W")
    if p['total_w'] <= args.power:
        print("✅ Power budget met")
    else:
        print(f"❌ Exceeds power budget by {p['total_w'] - args.power:.1f}W")
    return 0 if p['total_w'] <= args.power else 1


def cmd_estimate(args):
    """Estimate die size, cost, yield."""
    # Import from mask-locked repo
    sys.path.insert(0, "/tmp/mask-locked-inference-chip/src")
    import tapeout_planner as tp
    
    process_map = {"28nm": tp.Foundry.SMIC_28, "40nm": tp.Foundry.SMIC_40,
                   "65nm": tp.Foundry.GF_65, "130nm": tp.Foundry.SKYWATER_130}
    die_sizes = {"1b": 48, "3b": 48, "7b": 100, "13b": 200}
    
    die_mm2 = die_sizes.get(args.model, 48)
    spec = tp.ChipSpec(name=args.model, die_area_mm2=die_mm2,
                      foundry=process_map.get(args.process, tp.Foundry.SMIC_28),
                      package=tp.PackageType.QFN, target_volume=10000,
                      yield_pct=args.yield_pct / 100)
    planner = tp.TapeoutPlanner(spec)
    report = planner.full_report()
    
    print(f"Estimate for {args.model} ({die_mm2}mm2, {args.process}, {args.yield_pct}% yield):")
    print(f"  NRE: ${report['nre']['total_nre_k']}K")
    print(f"  Per unit: ${report['per_unit']['total_cost']:.3f}")
    print(f"  Year 1 (10K units): ${report['annual']['total_year1_k']}K")
    print(f"  Break-even: {report['annual']['break_even_units']:,} units")
    print(f"  Timeline: {report['timeline']['total_months']} months")
    return 0


def cmd_simulate(args):
    """Simulate chip inference."""
    print(f"Simulating {args.model} with prompt: {args.prompt}")
    from sdk import FrozenIntelligenceSDK
    sdk = FrozenIntelligenceSDK()
    result = sdk.generate(args.prompt, max_tokens=args.tokens, stream=False)
    print(f"\nGenerated ({result.generated_tokens} tokens, {result.elapsed_ms:.0f}ms):")
    print(f"  {result.text}")
    print(f"  Speed: {result.tokens_per_second:.0f} tokens/sec")
    return 0


def cmd_benchmark(args):
    """Run comprehensive benchmarks."""
    print(f"Running {args.iterations} iterations...")
    from sdk import FrozenIntelligenceSDK
    from chip_verifier import ChipVerifier
    from weight_compiler import WeightCompiler
    
    sdk = FrozenIntelligenceSDK()
    verifier = ChipVerifier()
    compiler = WeightCompiler()
    
    results = {}
    
    # Compilation benchmark
    start = time.time()
    for i in range(min(args.iterations, 5)):
        weights = compiler.generate_model(f"test-{i}", hidden_dim=512, num_layers=4)
        chip = compiler.compile(f"test-{i}", weights)
    results["compile_ms"] = (time.time() - start) * 1000 / min(args.iterations, 5)
    
    # Verification benchmark
    start = time.time()
    checks = verifier.run_timing_checks()
    results["verify_ms"] = (time.time() - start) * 1000
    
    # Inference benchmark
    bench = sdk.benchmark(num_prompts=min(args.iterations, 10), max_tokens=32)
    results.update(bench)
    
    print("Benchmark Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.1f}")
        else:
            print(f"  {k:20s}: {v}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Frozen Intelligence CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # compile
    p_compile = subparsers.add_parser("compile", help="Compile model to chip binary")
    p_compile.add_argument("--model", default="qwen3.5-3b", help="Model name")
    p_compile.add_argument("--precision", choices=["int4", "int8", "fp32"], default="int4")
    p_compile.add_argument("--output", help="Output binary file")
    
    # verify
    p_verify = subparsers.add_parser("verify", help="Verify chip constraints")
    p_verify.add_argument("--chip", choices=["scout", "messenger", "navigator", "captain"], default="messenger")
    p_verify.add_argument("--clock", type=int, default=500, help="Clock MHz")
    p_verify.add_argument("--power", type=float, default=3.0, help="Power budget W")
    
    # estimate
    p_estimate = subparsers.add_parser("estimate", help="Estimate die size and cost")
    p_estimate.add_argument("--model", choices=["1b", "3b", "7b", "13b"], default="3b")
    p_estimate.add_argument("--process", choices=["28nm", "40nm", "65nm", "130nm"], default="28nm")
    p_estimate.add_argument("--yield_pct", type=float, default=85.0, help="Yield percentage")
    
    # simulate
    p_simulate = subparsers.add_parser("simulate", help="Simulate chip inference")
    p_simulate.add_argument("--model", choices=["scout", "messenger"], default="messenger")
    p_simulate.add_argument("--prompt", default="Hello", help="Input prompt")
    p_simulate.add_argument("--tokens", type=int, default=64, help="Tokens to generate")
    
    # benchmark
    p_benchmark = subparsers.add_parser("benchmark", help="Run comprehensive benchmarks")
    p_benchmark.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    
    args = parser.parse_args()
    
    # Dispatch to command
    commands = {
        "compile": cmd_compile,
        "verify": cmd_verify,
        "estimate": cmd_estimate,
        "simulate": cmd_simulate,
        "benchmark": cmd_benchmark,
    }
    
    try:
        return commands[args.command](args)
    except ImportError as e:
        print(f"Error: Module not found. Make sure you're in the frozen-intelligence directory.")
        print(f"Details: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
