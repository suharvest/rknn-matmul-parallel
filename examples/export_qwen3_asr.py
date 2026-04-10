#!/usr/bin/env python3
"""
Export Qwen3-ASR-0.6B decoder weights for matmul_decoder.

This script demonstrates how to adapt a new model for use with
the matmul_decoder library.

Usage:
    python export_qwen3_asr.py /path/to/qwen3-asr-0.6b /output/dir
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_qwen3_asr(model_path: str, output_dir: str, quant_type: str = "int4"):
    """
    Export Qwen3-ASR decoder weights to matmul_decoder format.

    Output structure:
        output_dir/
        ├── config.json           # Model architecture config
        ├── embeddings.npy        # [vocab_size, hidden_dim] FP32
        ├── lm_head.npy           # [hidden_dim, vocab_size] FP32 (optional if tied)
        └── layers/
            ├── layer_00.npz      # {q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, input_norm, post_attn_norm}
            └── ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layers").mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    config = model.config

    # Export config
    decoder_config = {
        "name": config.name_or_path.split("/")[-1],
        "hidden_dim": config.hidden_size,
        "num_q_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
        "ffn_dim": config.intermediate_size,
        "num_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "rms_eps": config.rms_norm_eps,
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(decoder_config, f, indent=2)
    print(f"Exported config: {decoder_config}")

    # Export embeddings
    embed_weight = model.model.embed_tokens.weight.detach().numpy()
    np.save(output_dir / "embeddings.npy", embed_weight.astype(np.float32))
    print(f"Exported embeddings: {embed_weight.shape}")

    # Export lm_head (if not tied)
    if not decoder_config["tie_word_embeddings"]:
        lm_head = model.lm_head.weight.detach().numpy()
        np.save(output_dir / "lm_head.npy", lm_head.astype(np.float32))
        print(f"Exported lm_head: {lm_head.shape}")

    # Export layers
    for i, layer in enumerate(model.model.layers):
        layer_dict = {
            # Attention weights
            "q_proj": layer.self_attn.q_proj.weight.detach().numpy(),
            "k_proj": layer.self_attn.k_proj.weight.detach().numpy(),
            "v_proj": layer.self_attn.v_proj.weight.detach().numpy(),
            "o_proj": layer.self_attn.o_proj.weight.detach().numpy(),
            # FFN weights
            "gate_proj": layer.mlp.gate_proj.weight.detach().numpy(),
            "up_proj": layer.mlp.up_proj.weight.detach().numpy(),
            "down_proj": layer.mlp.down_proj.weight.detach().numpy(),
            # Normalization weights
            "input_norm": layer.input_layernorm.weight.detach().numpy(),
            "post_attn_norm": layer.post_attention_layernorm.weight.detach().numpy(),
        }

        # Convert to FP16 for storage efficiency
        for key in layer_dict:
            layer_dict[key] = layer_dict[key].astype(np.float16)

        np.savez_compressed(output_dir / "layers" / f"layer_{i:02d}.npz", **layer_dict)
        if i == 0 or (i + 1) % 10 == 0:
            print(f"Exported layer {i + 1}/{decoder_config['num_layers']}")

    print(f"\nExport complete! Weights saved to {output_dir}")

    # Print estimated size
    total_params = 0
    for f in output_dir.rglob("*.np*"):
        arr = np.load(f)
        if hasattr(arr, "keys"):
            for k in arr.keys():
                total_params += arr[k].size
        else:
            total_params += arr.size
    print(f"Total parameters: {total_params / 1e6:.1f}M")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR decoder weights")
    parser.add_argument("model_path", help="Path to Qwen3-ASR model")
    parser.add_argument("output_dir", help="Output directory for exported weights")
    parser.add_argument("--quant", default="int4", choices=["fp16", "int4", "int8"],
                        help="Quantization type (default: int4)")
    args = parser.parse_args()

    export_qwen3_asr(args.model_path, args.output_dir, args.quant)


if __name__ == "__main__":
    main()