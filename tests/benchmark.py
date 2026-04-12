#!/usr/bin/env python3
"""Performance benchmark: measures decode speed for all model variants.

Auto-discovers model directories and benchmarks each quantization type.
Outputs markdown table and JSON results.

Usage:
    python3 tests/benchmark.py                           # Auto-discover, benchmark all
    python3 tests/benchmark.py --quant fp16 --quant int4 # Specific types
    python3 tests/benchmark.py --steps 100               # More steps for stable numbers
    python3 tests/benchmark.py --rkllm /path/to/model.rkllm  # Compare with RKLLM

Exit code: always 0 (benchmark, not pass/fail).
"""
import argparse, json, os, sys, time
import numpy as np


def get_device_info():
    """Detect Rockchip SoC model."""
    info = {"soc": "unknown", "npu_cores": 0}
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            compat = f.read().decode("utf-8", errors="ignore")
        if "rk3576" in compat:
            info["soc"] = "RK3576"
            info["npu_cores"] = 2
        elif "rk3588" in compat:
            info["soc"] = "RK3588"
            info["npu_cores"] = 3
    except FileNotFoundError:
        pass

    # Runtime version
    try:
        import subprocess
        ver = subprocess.check_output(
            'strings /usr/lib/librknnrt.so 2>/dev/null | grep "librknnrt version"',
            shell=True, text=True
        ).strip()
        info["runtime"] = ver
    except Exception:
        info["runtime"] = "unknown"

    return info


def discover_models(base_dir):
    """Auto-discover model directories and their quant types."""
    models = {}
    name_map = {
        "matmul": "fp16",
        "matmul_fp16": "fp16",
        "matmul_w4a16": "int4",
        "matmul_int4": "int4",
        "matmul_w8a16": "int8",
        "matmul_int8": "int8",
        "matmul_w4a16_g128": "int4_g128",
    }
    for subdir in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, subdir)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
            quant = name_map.get(subdir)
            if quant:
                models[quant] = full
    return models


def benchmark_one(model_dir, quant_type, exec_mode, n_warmup, n_steps):
    """Benchmark a single configuration. Returns dict or None on error."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import matmul_decoder as md

    try:
        t0 = time.time()
        decoder = md.MatmulDecoder(
            model_dir=model_dir,
            max_seq_len=max(128, n_warmup + n_steps + 10),
            quant_type=quant_type,
            exec_mode=exec_mode,
        )
        load_time = time.time() - t0
    except Exception as e:
        return {"error": str(e)}

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    decoder.clear_kv_cache()
    token_id = 151644
    logits = decoder.step(token_id=token_id)
    next_token = int(logits.argmax())

    for _ in range(n_warmup):
        logits = decoder.step(token_id=next_token)
        next_token = int(logits.argmax())

    times = []
    for _ in range(n_steps):
        t_start = time.perf_counter()
        logits = decoder.step(token_id=next_token)
        t_end = time.perf_counter()
        next_token = int(logits.argmax())
        times.append((t_end - t_start) * 1000)

    times = np.array(times)
    stats = decoder.stats
    del decoder

    return {
        "quant": quant_type,
        "mode": exec_mode,
        "model": config.get("name", "unknown"),
        "load_s": round(load_time, 1),
        "avg_ms": round(float(np.mean(times)), 1),
        "p50_ms": round(float(np.median(times)), 1),
        "p99_ms": round(float(np.percentile(times, 99)), 1),
        "std_ms": round(float(np.std(times)), 1),
        "tok_s": round(float(1000.0 / np.mean(times)), 1),
        "matmul_ms": round(stats["matmul_ms"], 1),
        "rebind_ms": round(stats["rebind_ms"], 1),
        "lm_head_ms": round(stats["lm_head_ms"], 1),
        "cpu_ops_ms": round(stats["cpu_ops_ms"], 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-base", default=None, help="Base model directory")
    parser.add_argument("--quant", action="append", default=None, help="Quant types (repeatable)")
    parser.add_argument("--mode", default="single_core", help="Exec mode: single_core or dual_core")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rkllm", default=None, help="RKLLM model path for comparison")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Auto-detect model base
    model_base = args.model_base
    if not model_base:
        ref_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_data", "config.json")
        if os.path.exists(ref_cfg):
            with open(ref_cfg) as f:
                model_base = os.path.dirname(json.load(f).get("model_dir", ""))
        if not model_base:
            model_base = "/home/cat/qwen3-asr-models/decoder"

    device = get_device_info()
    models = discover_models(model_base)

    quant_types = args.quant or list(models.keys())
    quant_types = [q for q in quant_types if q in models]

    print(f"{'='*60}")
    print(f"Performance Benchmark — {device['soc']} ({device['npu_cores']} NPU cores)")
    print(f"{'='*60}")
    print(f"  {device.get('runtime', '')}")
    print(f"  mode={args.mode}, warmup={args.warmup}, steps={args.steps}")
    print(f"  models: {', '.join(quant_types)}")

    results = []
    for quant in quant_types:
        print(f"\n  Benchmarking {quant.upper()}...", end="", flush=True)
        r = benchmark_one(models[quant], quant, args.mode, args.warmup, args.steps)
        if "error" in r:
            print(f" ERROR: {r['error']}")
        else:
            print(f" {r['avg_ms']}ms/tok ({r['tok_s']} tok/s)")
            results.append(r)

    # Markdown table
    if results:
        print(f"\n## Results — {device['soc']}, {args.steps} steps\n")
        print("| Quant | ms/tok (avg) | ms/tok (p50) | tok/s | matmul | rebind | lm_head | cpu_ops |")
        print("|-------|-------------|-------------|-------|--------|--------|---------|---------|")
        for r in results:
            print(f"| {r['quant']:>5} | {r['avg_ms']:>11.1f} | {r['p50_ms']:>11.1f} | "
                  f"{r['tok_s']:>5.1f} | {r['matmul_ms']:>6.1f} | {r['rebind_ms']:>6.1f} | "
                  f"{r['lm_head_ms']:>7.1f} | {r['cpu_ops_ms']:>7.1f} |")

    # Save JSON
    output = {"device": device, "config": {"mode": args.mode, "warmup": args.warmup, "steps": args.steps}, "results": results}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
