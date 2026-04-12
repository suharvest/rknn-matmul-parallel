#!/usr/bin/env python3
"""
Qwen3-ASR on RK3576 using matmul decoder (no RKLLM needed).

Standalone example from rknn-matmul-parallel.
Uses the matmul_decoder C extension for autoregressive decoding,
avoiding the RKLLM/RKNN NPU conflict on RK3576.

Usage:
    cd examples/qwen3_asr
    python run_asr.py /path/to/audio.wav

    # With options:
    python run_asr.py audio.wav --model-dir /path/to/models --language English

    # Auto-detect language:
    python run_asr.py audio.wav --language auto
"""

import argparse
import sys
import os
import time

# Add rknn-matmul-parallel root to path for matmul_decoder C extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import Qwen3ASREngine


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR transcription using matmul decoder")
    parser.add_argument("audio", help="Path to audio file (wav, mp3, flac, etc.)")
    parser.add_argument("--model-dir", default="/home/cat/models/qwen3-asr",
                        help="Model directory (default: /home/cat/models/qwen3-asr)")
    parser.add_argument("--language", default="Chinese",
                        help="Language hint: Chinese, English, auto, etc. "
                             "(default: Chinese)")
    parser.add_argument("--platform", default="rk3576",
                        choices=["rk3576", "rk3588"],
                        help="Target platform (default: rk3576)")
    parser.add_argument("--exec-mode", default="dual_core",
                        choices=["single_core", "dual_core"],
                        help="NPU execution mode (default: dual_core)")
    parser.add_argument("--chunk-size", type=float, default=30.0,
                        help="Audio chunk size in seconds (default: 30)")
    parser.add_argument("--max-new-tokens", type=int, default=500,
                        help="Max tokens to generate per chunk (default: 500)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print the final transcription")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    language = args.language if args.language != "auto" else None

    engine = Qwen3ASREngine(
        model_dir=args.model_dir,
        platform=args.platform,
        decoder_exec_mode=args.exec_mode,
        max_new_tokens=args.max_new_tokens,
        verbose=not args.quiet,
    )

    try:
        result = engine.transcribe(
            args.audio,
            language=language,
            chunk_size=args.chunk_size,
        )

        if args.quiet:
            print(result["text"])
        else:
            print(f"\n{'='*60}")
            print(f"Language: {result['language']}")
            print(f"Text: {result['text']}")
            print(f"\nPerformance:")
            stats = result["stats"]
            print(f"  Audio duration: {stats['audio_s']:.1f}s")
            print(f"  Encoder time:   {stats['enc_ms']:.0f}ms")
            print(f"  Decoder time:   {stats['llm_ms']:.0f}ms")
            print(f"  Wall time:      {stats['wall_ms']:.0f}ms")
            print(f"  RTF:            {stats['rtf']:.3f}")
            print(f"  Chunks:         {stats['chunks']}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
