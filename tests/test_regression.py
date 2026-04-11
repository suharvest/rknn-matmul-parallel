#!/usr/bin/env python3
"""Regression test: compare C/NPU decoder output against numpy reference.

Prerequisites:
    1. Generate reference data first:
       python3 tests/generate_reference.py --model-dir /path/to/weights --output-dir tests/reference_data
    2. Build Python binding:
       make python

Usage:
    python3 tests/test_regression.py \
        --model-dir /path/to/weights \
        --reference-dir tests/reference_data

Exit code 0 = all checks pass, 1 = regression detected.
"""

import argparse
import json
import os
import sys
import numpy as np


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


def check(name, actual, expected, cos_threshold=0.99, atol=None):
    """Compare two arrays, return (pass, cosine, details)."""
    cos = cosine_sim(actual, expected)
    passed = cos >= cos_threshold
    details = f"cosine={cos:.6f} (threshold={cos_threshold})"
    if atol is not None:
        max_diff = np.max(np.abs(actual - expected))
        details += f", max_diff={max_diff:.6f} (atol={atol})"
        if max_diff > atol:
            passed = False
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {details}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Regression test for matmul decoder")
    parser.add_argument("--model-dir", required=True, help="Path to matmul weights")
    parser.add_argument("--reference-dir", default="tests/reference_data", help="Path to numpy reference .npy files")
    parser.add_argument("--cos-threshold", type=float, default=0.99, help="Minimum cosine similarity to pass")
    parser.add_argument("--exec-mode", default="single_core", choices=["single_core", "dual_core"])
    args = parser.parse_args()

    ref_dir = args.reference_dir
    if not os.path.isdir(ref_dir):
        print(f"ERROR: Reference directory not found: {ref_dir}")
        print("Run generate_reference.py first.")
        sys.exit(1)

    # Load test config
    with open(os.path.join(ref_dir, "config.json")) as f:
        test_cfg = json.load(f)
    token_id = test_cfg["token_id"]
    expected_top1 = test_cfg["expected_top1"]
    expected_top5 = test_cfg["expected_top5"]

    print(f"Regression test: token_id={token_id}, expected_top1={expected_top1}")
    print(f"Model: {args.model_dir}")
    print(f"Reference: {ref_dir}")
    print()

    # Import C decoder
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import matmul_decoder as md
    except ImportError as e:
        print(f"ERROR: Cannot import matmul_decoder: {e}")
        print("Build with: make python")
        sys.exit(1)

    # Create decoder
    print("Creating decoder...")
    decoder = md.MatmulDecoder(
        model_dir=args.model_dir,
        max_seq_len=64,
        quant_type="fp16",
        exec_mode=args.exec_mode,
    )
    print()

    # Run one step — step() returns logits array, step_get_token() returns int
    print("Running step with token_id=%d..." % token_id)
    decoder.clear_kv_cache()
    logits = np.array(decoder.step(token_id=token_id))  # [vocab_size]
    predicted = int(np.argmax(logits))
    logits_available = True
    print(f"Predicted token: {predicted}")
    print(f"Logits: std={logits.std():.4f}, min={logits.min():.4f}, max={logits.max():.4f}")
    print()

    # === Checks ===
    all_passed = True
    print("=== Regression Checks ===")

    # Check 1: Top-1 token
    if predicted == expected_top1:
        print(f"  [PASS] top1_token: {predicted} == {expected_top1}")
    else:
        print(f"  [FAIL] top1_token: {predicted} != {expected_top1}")
        all_passed = False

    # Check 2: Logits cosine (if available)
    if logits_available:
        ref_top_idx = np.load(os.path.join(ref_dir, "logits_top100_idx.npy"))
        ref_top_vals = np.load(os.path.join(ref_dir, "logits_top100.npy"))

        # Compare logits at the top-100 reference positions
        actual_top_vals = logits[ref_top_idx]
        if not check("logits_top100", actual_top_vals, ref_top_vals,
                      cos_threshold=args.cos_threshold):
            all_passed = False

        # Check top-5 overlap
        actual_top5 = np.argsort(logits)[::-1][:5].tolist()
        overlap = len(set(actual_top5) & set(expected_top5))
        if overlap >= 4:
            print(f"  [PASS] top5_overlap: {overlap}/5 (actual={actual_top5})")
        else:
            print(f"  [FAIL] top5_overlap: {overlap}/5 (actual={actual_top5}, expected={expected_top5})")
            all_passed = False

    # Check 3: Layer intermediates (if saved)
    for layer_file in sorted(os.listdir(ref_dir)):
        if layer_file.startswith("layer_") and layer_file.endswith(".npy"):
            # These can only be checked if the C library exposes debug dumps
            pass  # Reserved for future debug mode

    print()
    if all_passed:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("REGRESSION DETECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
