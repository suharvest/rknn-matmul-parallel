#!/usr/bin/env python3
"""Generate numpy reference outputs for regression testing.

Runs a full forward pass in pure numpy (no NPU) and saves intermediate
values at each layer + final logits. These serve as ground truth for
verifying the C/NPU matmul decoder.

Usage:
    python3 tests/generate_reference.py \
        --model-dir /path/to/matmul_weights \
        --output-dir tests/reference_data \
        --token-id 151644

Output:
    reference_data/
    ├── input_embedding.npy       # [hidden_dim]
    ├── layer_00_after_norm.npy   # [hidden_dim]
    ├── layer_00_q_proj.npy       # [q_dim]
    ├── layer_00_after_attn.npy   # [hidden_dim]  (after o_proj + residual)
    ├── layer_00_after_ffn.npy    # [hidden_dim]  (after FFN + residual)
    ├── ...
    ├── layer_27_after_ffn.npy
    ├── final_normed.npy          # [hidden_dim]
    ├── logits_top100.npy         # [100] top-100 logit values
    ├── logits_top100_idx.npy     # [100] top-100 token indices
    └── config.json               # test configuration
"""

import argparse
import json
import os
import numpy as np


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def rope_interleaved(x_heads, pos, head_dim, theta=1000000.0):
    """Apply interleaved RoPE to [num_heads, head_dim] at given position."""
    half = head_dim // 2
    out = x_heads.copy()
    for i in range(half):
        freq = 1.0 / (theta ** (2.0 * i / head_dim))
        angle = pos * freq
        c, s = np.cos(angle), np.sin(angle)
        for h in range(out.shape[0]):
            x0 = out[h, 2 * i]
            x1 = out[h, 2 * i + 1]
            out[h, 2 * i] = x0 * c - x1 * s
            out[h, 2 * i + 1] = x1 * c + x0 * s
    return out


def matmul_fp16(x_fp32, w_fp16):
    """Simulate FP16 matmul: cast input to FP16, multiply, return FP32."""
    return (x_fp32.astype(np.float16) @ w_fp16).astype(np.float32)


def forward_layer(x, layer_dir, pos, config, save_prefix=None, out_dir=None):
    """Run one transformer layer, optionally saving intermediates."""
    hidden_dim = config["hidden_dim"]
    num_q_heads = config["num_q_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    ffn_dim = config["ffn_dim"]
    eps = config.get("rms_eps", 1e-6)
    theta = config.get("rope_theta", 1000000.0)
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    def save(name, arr):
        if save_prefix and out_dir:
            np.save(os.path.join(out_dir, f"{save_prefix}_{name}.npy"), arr)

    # 1. Input norm
    norm_w = np.fromfile(f"{layer_dir}/input_norm.bin", dtype=np.float32)
    normed = rms_norm(x, norm_w, eps)
    save("after_norm", normed)

    # 2. QKV projections
    w_q = np.fromfile(f"{layer_dir}/q_proj.bin", dtype=np.float16).reshape(hidden_dim, q_dim)
    w_k = np.fromfile(f"{layer_dir}/k_proj.bin", dtype=np.float16).reshape(hidden_dim, kv_dim)
    w_v = np.fromfile(f"{layer_dir}/v_proj.bin", dtype=np.float16).reshape(hidden_dim, kv_dim)

    q = matmul_fp16(normed, w_q)  # [q_dim]
    k = matmul_fp16(normed, w_k)  # [kv_dim]
    v = matmul_fp16(normed, w_v)  # [kv_dim]
    save("q_proj", q)
    save("k_proj", k)
    save("v_proj", v)

    # 3. QK norm (per-head RMSNorm)
    if config.get("has_qk_norm", False):
        q_norm_w = np.fromfile(f"{layer_dir}/q_norm.bin", dtype=np.float32)
        k_norm_w = np.fromfile(f"{layer_dir}/k_norm.bin", dtype=np.float32)
        q = q.reshape(num_q_heads, head_dim)
        k = k.reshape(num_kv_heads, head_dim)
        for h in range(num_q_heads):
            q[h] = rms_norm(q[h], q_norm_w, eps)
        for h in range(num_kv_heads):
            k[h] = rms_norm(k[h], k_norm_w, eps)
    else:
        q = q.reshape(num_q_heads, head_dim)
        k = k.reshape(num_kv_heads, head_dim)

    # 4. RoPE
    q = rope_interleaved(q, pos, head_dim, theta)
    k = rope_interleaved(k, pos, head_dim, theta)
    save("after_rope_q", q.reshape(-1))

    # 5. Attention (single position: softmax over 1 token = identity, output = V)
    v_heads = v.reshape(num_kv_heads, head_dim)
    groups = num_q_heads // num_kv_heads
    attn_out = np.zeros(q_dim, dtype=np.float32)
    for qh in range(num_q_heads):
        kvh = qh // groups
        # With single token, attention score is trivially 1.0
        attn_out[qh * head_dim:(qh + 1) * head_dim] = v_heads[kvh]
    save("attn_out", attn_out)

    # 6. O projection + residual
    w_o = np.fromfile(f"{layer_dir}/o_proj.bin", dtype=np.float16).reshape(q_dim, hidden_dim)
    o_out = matmul_fp16(attn_out, w_o)
    hidden = x + o_out  # residual
    save("after_attn", hidden)

    # 7. Post-attention norm
    post_w = np.fromfile(f"{layer_dir}/post_attn_norm.bin", dtype=np.float32)
    normed2 = rms_norm(hidden, post_w, eps)

    # 8. FFN (SwiGLU)
    w_gate = np.fromfile(f"{layer_dir}/gate_proj.bin", dtype=np.float16).reshape(hidden_dim, ffn_dim)
    w_up = np.fromfile(f"{layer_dir}/up_proj.bin", dtype=np.float16).reshape(hidden_dim, ffn_dim)
    w_down = np.fromfile(f"{layer_dir}/down_proj.bin", dtype=np.float16).reshape(ffn_dim, hidden_dim)

    gate = matmul_fp16(normed2, w_gate)
    up = matmul_fp16(normed2, w_up)
    ffn_act = silu(gate) * up
    ffn_out = matmul_fp16(ffn_act, w_down)

    hidden = hidden + ffn_out  # residual
    save("after_ffn", hidden)

    return hidden


def main():
    parser = argparse.ArgumentParser(description="Generate numpy reference for regression testing")
    parser.add_argument("--model-dir", required=True, help="Path to matmul weights directory")
    parser.add_argument("--output-dir", default="tests/reference_data", help="Output directory for reference .npy files")
    parser.add_argument("--token-id", type=int, default=151644, help="Input token ID (default: <|im_start|>)")
    parser.add_argument("--save-all-layers", action="store_true", help="Save intermediates for all layers (not just 0 and last)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    print(f"Config: hidden={config['hidden_dim']}, heads={config['num_q_heads']}/{config['num_kv_heads']}, "
          f"head_dim={config['head_dim']}, layers={config['num_layers']}, ffn={config['ffn_dim']}")

    # Load embedding
    emb_path = os.path.join(args.model_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
    else:
        emb_bin = os.path.join(args.model_dir, "embeddings.bin")
        emb = np.fromfile(emb_bin, dtype=np.float32).reshape(-1, config["hidden_dim"])

    x = emb[args.token_id].astype(np.float32)
    print(f"Input token {args.token_id}: embedding std={x.std():.6f}")
    np.save(os.path.join(args.output_dir, "input_embedding.npy"), x)

    # Run all layers
    num_layers = config["num_layers"]
    for i in range(num_layers):
        layer_dir = os.path.join(args.model_dir, "layers", f"layer_{i:02d}")
        if not os.path.isdir(layer_dir):
            layer_dir = os.path.join(args.model_dir, f"layer_{i:02d}")

        # Save intermediates for layer 0, last layer, and optionally all
        save_prefix = None
        if i == 0 or i == num_layers - 1 or args.save_all_layers:
            save_prefix = f"layer_{i:02d}"

        x = forward_layer(x, layer_dir, pos=0, config=config,
                          save_prefix=save_prefix, out_dir=args.output_dir)

        if i % 4 == 0 or i == num_layers - 1:
            print(f"  Layer {i:2d}: hidden std={x.std():.6f}, mean={x.mean():.6f}")

    # Final norm
    final_norm_path = os.path.join(args.model_dir, "final_norm.bin")
    if os.path.exists(final_norm_path):
        final_w = np.fromfile(final_norm_path, dtype=np.float32)
    else:
        # Fallback: try model_norm.bin
        final_w = np.fromfile(os.path.join(args.model_dir, "model_norm.bin"), dtype=np.float32)

    eps = config.get("rms_eps", 1e-6)
    final_normed = rms_norm(x, final_w, eps)
    np.save(os.path.join(args.output_dir, "final_normed.npy"), final_normed)

    # Logits: hidden @ embeddings.T (tied weights)
    logits = final_normed @ emb.T  # [vocab_size]
    top_idx = np.argsort(logits)[::-1][:100]
    top_vals = logits[top_idx]

    np.save(os.path.join(args.output_dir, "logits_top100.npy"), top_vals)
    np.save(os.path.join(args.output_dir, "logits_top100_idx.npy"), top_idx)

    print(f"\nFinal logits: std={logits.std():.4f}, top1={top_idx[0]} (val={top_vals[0]:.4f})")
    print(f"Top 5 tokens: {top_idx[:5].tolist()}")
    print(f"Top 5 values: {top_vals[:5].tolist()}")

    # Save test config
    test_config = {
        "token_id": args.token_id,
        "model_dir": args.model_dir,
        "num_layers": num_layers,
        "expected_top1": int(top_idx[0]),
        "expected_top5": top_idx[:5].tolist(),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(test_config, f, indent=2)

    print(f"\nReference data saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
