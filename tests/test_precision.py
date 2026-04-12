#!/usr/bin/env python3
"""Precision regression test: compare decoder outputs against numpy reference.

Automatically discovers all model variants (fp16, int4, int8, etc.) and tests each.
Designed as a permanent regression test — run after every code change.

Usage:
    python3 test_precision.py                    # Auto-discover and test all models
    python3 test_precision.py --quant fp16       # Test specific quant types
    python3 test_precision.py --steps 5          # Multi-step test
    python3 test_precision.py --json             # Machine-readable output
    python3 test_precision.py --model-base /path # Custom model directory

Exit code: 0 if FP16 passes thresholds, 1 otherwise.
"""
import sys, os, json, argparse, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

# ─── Thresholds ───
# FP16 must meet these (implementation correctness)
FP16_COS_THRESHOLD = 0.99
FP16_TOP100_THRESHOLD = 70
# Quantized models: reported but no hard pass/fail (precision varies by model)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def analyze_vs_ref(logits, ref_logits, ref_indices, ref_top1, ref_top5):
    """Compare logits against numpy reference."""
    top1 = int(logits.argmax())
    top5 = np.argsort(logits)[::-1][:5].tolist()
    top100 = np.argsort(logits)[::-1][:100].tolist()

    our_vals = logits[ref_indices]
    cos_top100 = cosine_sim(our_vals, ref_logits)
    ovlp5 = len(set(top5) & set(ref_top5))
    ovlp100 = len(set(top100) & set(ref_indices.tolist()))

    return {
        "top1": top1,
        "top1_match": top1 == ref_top1,
        "top5": top5,
        "top5_overlap": ovlp5,
        "top100_overlap": ovlp100,
        "cos_top100": cos_top100,
        "logits_min": float(logits.min()),
        "logits_max": float(logits.max()),
        "logits_std": float(logits.std()),
    }


def discover_models(base_dir):
    """Auto-discover model directories and their quant types."""
    models = {}
    # Standard naming: matmul (fp16), matmul_w4a16 (int4), matmul_w8a16 (int8), etc.
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
        if not os.path.isdir(full):
            continue
        config_path = os.path.join(full, "config.json")
        if not os.path.exists(config_path):
            continue
        quant = name_map.get(subdir)
        if quant:
            models[quant] = full
    return models


def run_test(model_dir, quant_type, ref_data, args):
    """Run decoder and return metrics dict."""
    import matmul_decoder as md

    try:
        d = md.MatmulDecoder(
            model_dir=model_dir,
            max_seq_len=max(16, args.steps + 2),
            quant_type=quant_type,
            exec_mode="single_core",
            disable_npu_lm_head=args.disable_npu_lm_head,
        )
    except Exception as e:
        return {"error": str(e)}

    d.clear_kv_cache()
    token_id = ref_data["token_id"]

    t0 = time.time()
    for step in range(args.steps):
        logits = d.step(token_id=token_id)
    wall_ms = (time.time() - t0) * 1000

    metrics = analyze_vs_ref(
        logits,
        ref_data["logits_top100"],
        ref_data["logits_idx"],
        ref_data["top1"],
        ref_data["top5"],
    )
    stats = d.stats
    metrics["total_ms"] = stats["total_ms"]
    metrics["matmul_ms"] = stats["matmul_ms"]
    metrics["rebind_ms"] = stats["rebind_ms"]
    metrics["lm_head_ms"] = stats["lm_head_ms"]
    metrics["wall_ms"] = wall_ms
    metrics["model_dir"] = model_dir

    del d
    return metrics


def load_reference(ref_dir):
    """Load numpy reference data."""
    with open(os.path.join(ref_dir, "config.json")) as f:
        cfg = json.load(f)
    return {
        "token_id": cfg["token_id"],
        "top1": cfg["expected_top1"],
        "top5": cfg["expected_top5"],
        "logits_top100": np.load(os.path.join(ref_dir, "logits_top100.npy")),
        "logits_idx": np.load(os.path.join(ref_dir, "logits_top100_idx.npy")).astype(int),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-base", default=None,
                        help="Base directory containing model subdirs")
    parser.add_argument("--ref-dir", default=None,
                        help="Reference data directory")
    parser.add_argument("--quant", action="append", default=None,
                        help="Quant types to test (can repeat). Default: auto-discover all.")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of decoder steps")
    parser.add_argument("--disable-npu-lm-head", action="store_true",
                        help="Force CPU FP16 lm_head")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON results")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output (pass/fail only)")
    args = parser.parse_args()

    # Defaults
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = args.ref_dir or os.path.join(script_dir, "reference_data")
    model_base = args.model_base

    # Auto-detect model base from reference config
    if not model_base:
        ref_cfg_path = os.path.join(ref_dir, "config.json")
        if os.path.exists(ref_cfg_path):
            with open(ref_cfg_path) as f:
                cfg = json.load(f)
            if "model_dir" in cfg:
                model_base = os.path.dirname(cfg["model_dir"])
    if not model_base:
        model_base = "/home/cat/qwen3-asr-models/decoder"

    # Load reference
    ref_data = load_reference(ref_dir)

    # Discover models
    all_models = discover_models(model_base)
    if args.quant:
        models = {q: all_models[q] for q in args.quant if q in all_models}
        missing = [q for q in args.quant if q not in all_models]
        if missing:
            print(f"Warning: models not found for: {', '.join(missing)}")
            print(f"Available: {', '.join(all_models.keys())}")
    else:
        models = all_models

    if not models:
        print(f"No models found in {model_base}")
        sys.exit(1)

    # Print header
    if not args.quiet:
        print(f"{'='*60}")
        print(f"Precision Regression Test")
        print(f"{'='*60}")
        print(f"  reference: top1={ref_data['top1']}, top5={ref_data['top5']}")
        print(f"  token_id={ref_data['token_id']}, steps={args.steps}")
        print(f"  models: {', '.join(models.keys())}")
        if args.disable_npu_lm_head:
            print(f"  NPU lm_head: DISABLED")

    # Run tests
    results = {}
    all_logits = {}

    # Always test fp16 first (baseline)
    ordered = sorted(models.keys(), key=lambda x: 0 if x == "fp16" else 1)

    for quant in ordered:
        model_dir = models[quant]
        if not args.quiet:
            print(f"\n  [{quant.upper()}] {model_dir}")

        metrics = run_test(model_dir, quant, ref_data, args)

        if "error" in metrics:
            if not args.quiet:
                print(f"    ERROR: {metrics['error']}")
            results[quant] = metrics
            continue

        results[quant] = metrics

        if not args.quiet:
            m = metrics
            match = "YES" if m["top1_match"] else "no"
            print(f"    top1={m['top1']} (ref={ref_data['top1']}, match={match})")
            print(f"    cos(top100) = {m['cos_top100']:.6f}")
            print(f"    top5 overlap  = {m['top5_overlap']}/5, top100 overlap = {m['top100_overlap']}/100")
            print(f"    range=[{m['logits_min']:.3f}, {m['logits_max']:.3f}], std={m['logits_std']:.3f}")
            print(f"    time = {m['total_ms']:.1f} ms (matmul={m['matmul_ms']:.1f}, lm_head={m['lm_head_ms']:.1f})")

    # Cross-comparison
    valid = {q: r for q, r in results.items() if "error" not in r}
    if len(valid) > 1 and not args.quiet:
        print(f"\n  {'─'*50}")
        keys = list(valid.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                # Compare at ref indices
                print(f"    {a} vs {b}: (see cos_top100 above)")

    # Summary table
    if len(valid) > 1 and not args.quiet:
        print(f"\n  {'─'*50}")
        print(f"  Summary:")
        header = f"  {'metric':>20}"
        for q in valid:
            header += f" {q.upper():>12}"
        print(header)
        print(f"  {'─'*20}" + f" {'─'*12}" * len(valid))
        for metric, fmt in [
            ("top1", ""), ("cos_top100", ".6f"),
            ("top5_overlap", ""), ("top100_overlap", ""),
            ("total_ms", ".1f"),
        ]:
            row = f"  {metric:>20}"
            for q in valid:
                v = valid[q][metric]
                if metric in ("top5_overlap",):
                    row += f" {v:>10}/5"
                elif metric in ("top100_overlap",):
                    row += f" {v:>8}/100"
                elif fmt:
                    row += f" {v:>12{fmt}}"
                else:
                    row += f" {v:>12}"

            print(row)

    # Pass/fail
    print(f"\n  {'='*50}")
    all_pass = True
    for q, m in results.items():
        if "error" in m:
            print(f"  {q.upper():>6}: ERROR — {m['error']}")
            if q == "fp16":
                all_pass = False
            continue

        if q == "fp16":
            ok = m["cos_top100"] > FP16_COS_THRESHOLD and m["top100_overlap"] >= FP16_TOP100_THRESHOLD
            status = "PASS" if ok else "FAIL"
            detail = f"cos={m['cos_top100']:.4f}>{FP16_COS_THRESHOLD}, top100={m['top100_overlap']}>={FP16_TOP100_THRESHOLD}"
            print(f"  {q.upper():>6}: {status} ({detail})")
            if not ok:
                all_pass = False
        else:
            grade = "GOOD" if m["cos_top100"] > 0.95 else "FAIR" if m["cos_top100"] > 0.8 else "POOR"
            print(f"  {q.upper():>6}: {grade} (cos={m['cos_top100']:.4f}, top100={m['top100_overlap']}/100, {m['total_ms']:.0f}ms)")

    if args.json:
        for q in results:
            for k, v in list(results[q].items()):
                if isinstance(v, (np.integer,)):
                    results[q][k] = int(v)
                elif isinstance(v, (np.floating,)):
                    results[q][k] = float(v)
                elif isinstance(v, np.ndarray):
                    results[q][k] = v.tolist()
        print(f"\n{json.dumps(results, indent=2)}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
