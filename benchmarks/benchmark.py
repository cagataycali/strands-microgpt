#!/usr/bin/env python3
"""MicroGPT Benchmark — v1 vs v2 Parameter Golf comparison.

Trains both v1 (baseline) and v2 (Parameter Golf) configurations on
the same dataset with identical hyperparameters, then compares:

  1. Final loss (bits per byte)
  2. Loss at equivalent parameter count
  3. Training speed (steps/sec)
  4. Parameter efficiency (loss / params)
  5. Generation quality (unique samples, avg length)

Outputs JSON to stdout for CI consumption.

Usage:
    python benchmarks/benchmark.py                    # default (200 steps)
    python benchmarks/benchmark.py --steps 500        # more training
    python benchmarks/benchmark.py --dataset path.txt # custom dataset
    python benchmarks/benchmark.py --output results.json
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request

# Add parent to path so we can import strands_microgpt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands_microgpt.engine import MicroGPT, Tokenizer

# ── Dataset ──────────────────────────────────────────────────────────────────

DATASET_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"


def load_dataset(path=None):
    """Load dataset from local file or download names.txt."""
    if path and os.path.exists(path):
        with open(path) as f:
            text = f.read()
    else:
        print("📥 Downloading dataset...", file=sys.stderr)
        text = urllib.request.urlopen(DATASET_URL).read().decode("utf-8")

    docs = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return docs


# ── Benchmark configs ────────────────────────────────────────────────────────

# Shared hyperparameters (fair comparison)
SHARED = {
    "n_embd": 16,
    "block_size": 16,
    "n_head": 4,
    "n_layer": 2,
    "seed": 42,
}

# v1: baseline (no Parameter Golf techniques)
V1_CONFIG = {
    **SHARED,
    "n_kv_head": 4,       # full KV heads (no GQA)
    "tie_embeddings": False,
    "use_rope": False,
    "use_bigram_hash": False,
    "logit_softcap": 0.0,  # disabled
    "use_leaky_relu": False,  # v1: plain ReLU²
    "use_resid_mix": False,   # v1: no residual mixing
    "use_smear_gate": False,  # v1: no SmearGate
    "weight_decay": 0.0,     # v1: no weight decay
    "warmup_steps": 0,       # v1: no warmup
}

# v2: all Parameter Golf techniques enabled
V2_CONFIG = {
    **SHARED,
    "n_kv_head": 2,        # GQA: half KV heads
    "tie_embeddings": True,
    "use_rope": True,
    "use_bigram_hash": False,  # keep fair — bigram adds extra params
    "logit_softcap": 30.0,
    "use_leaky_relu": False,  # v2: plain ReLU²
    "use_resid_mix": False,   # v2: no residual mixing
    "use_smear_gate": False,  # v2: no SmearGate
    "weight_decay": 0.0,     # v2: no weight decay
    "warmup_steps": 0,       # v2: no warmup
}

# v2+bigram: v2 with BigramHash (for completeness)
V2_BIGRAM_CONFIG = {
    **SHARED,
    "n_kv_head": 2,
    "tie_embeddings": True,
    "use_rope": True,
    "use_bigram_hash": True,
    "logit_softcap": 30.0,
    "use_leaky_relu": False,  # v2: plain ReLU²
    "use_resid_mix": False,   # v2: no residual mixing
    "use_smear_gate": False,  # v2: no SmearGate
    "weight_decay": 0.0,     # v2: no weight decay
    "warmup_steps": 0,       # v2: no warmup
}

# v3: Parameter Golf v3 — winning combo (WD=0.04)
# Weight decay is the #1 improvement from Parameter Golf for pure-Python:
# - Regularizes weights → better generalization
# - From SmearGate/OrthoInit/MuonWD submission: WD=0.04 was optimal
# - LeakyReLU² available but optional (scale-dependent)
V3_CONFIG = {
    **SHARED,
    "n_kv_head": 2,        # GQA: half KV heads
    "tie_embeddings": True,
    "use_rope": True,
    "use_bigram_hash": False,
    "logit_softcap": 30.0,
    "use_leaky_relu": False,  # scale-dependent, enable for larger models
    "leaky_relu_slope": 0.5,
    "use_smear_gate": False,
    "use_resid_mix": False,
    "weight_decay": 0.04,    # from Parameter Golf: optimal WD
    "warmup_steps": 0,
}

# v3+smeargate: v3 with SmearGate
V3_SMEARGATE_CONFIG = {
    **SHARED,
    "n_kv_head": 2,
    "tie_embeddings": True,
    "use_rope": True,
    "use_bigram_hash": False,
    "logit_softcap": 30.0,
    "use_leaky_relu": True,
    "leaky_relu_slope": 0.5,
    "use_smear_gate": True,
    "use_resid_mix": True,
    "weight_decay": 0.01,
    "warmup_steps": 10,
}


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(config, docs, tokenizer, num_steps, learning_rate, label):
    """Train a model and collect metrics."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {label}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    model = MicroGPT(vocab_size=tokenizer.vocab_size, **config)

    print(f"  Params: {model.num_params:,}", file=sys.stderr)
    print(f"  Techniques: {model.techniques}", file=sys.stderr)
    print(f"  Config: n_kv_head={config.get('n_kv_head', 4)}, "
          f"tied={config.get('tie_embeddings', False)}, "
          f"rope={config.get('use_rope', False)}", file=sys.stderr)

    # Train
    t0 = time.time()
    losses = model.train_on_docs(
        docs=docs,
        tokenizer=tokenizer,
        num_steps=num_steps,
        learning_rate=learning_rate,
        log_every=0,  # silent
    )
    train_time = time.time() - t0

    # Generate samples
    t_gen = time.time()
    samples = model.generate(
        tokenizer=tokenizer,
        num_samples=50,
        temperature=0.5,
    )
    gen_time = time.time() - t_gen  # noqa: F841 — kept for future reporting

    # Compute metrics
    final_loss = losses[-1] if losses else float("inf")

    # Loss at 25%, 50%, 75% checkpoints
    quarter = max(1, num_steps // 4)
    loss_25 = losses[quarter - 1] if len(losses) >= quarter else float("inf")
    loss_50 = losses[2 * quarter - 1] if len(losses) >= 2 * quarter else float("inf")
    loss_75 = losses[3 * quarter - 1] if len(losses) >= 3 * quarter else float("inf")

    # Convergence: steps to reach loss < threshold
    threshold = 3.0  # reasonable for names.txt
    steps_to_threshold = num_steps  # default: never reached
    for i, loss_val in enumerate(losses):
        if loss_val < threshold:
            steps_to_threshold = i + 1
            break

    # BPB (bits per byte) ≈ loss / ln(2)
    bpb = final_loss / math.log(2)

    # Generation quality
    nonempty = [s for s in samples if len(s) > 0]
    avg_length = sum(len(s) for s in nonempty) / max(len(nonempty), 1)
    unique_ratio = len(set(samples)) / max(len(samples), 1)

    # Parameter efficiency: lower loss × fewer params = better
    efficiency = final_loss * model.num_params

    result = {
        "label": label,
        "config": config,
        "num_params": model.num_params,
        "techniques": model.techniques,
        "training": {
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "train_time_sec": round(train_time, 2),
            "steps_per_sec": round(num_steps / train_time, 2),
        },
        "losses": {
            "final": round(final_loss, 6),
            "bpb": round(bpb, 6),
            "at_25pct": round(loss_25, 6),
            "at_50pct": round(loss_50, 6),
            "at_75pct": round(loss_75, 6),
            "steps_to_3.0": steps_to_threshold,
            "curve": [round(v, 4) for v in losses[::max(1, num_steps // 20)]],
        },
        "generation": {
            "num_samples": len(samples),
            "nonempty": len(nonempty),
            "unique_ratio": round(unique_ratio, 4),
            "avg_length": round(avg_length, 2),
            "examples": nonempty[:5],
        },
        "efficiency": {
            "loss_x_params": round(efficiency, 2),
            "bpb_per_1k_params": round(bpb / (model.num_params / 1000), 6),
        },
    }

    print(f"  Final loss: {final_loss:.4f} | BPB: {bpb:.4f} | "
          f"Time: {train_time:.1f}s | Efficiency: {efficiency:.0f}",
          file=sys.stderr)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MicroGPT Benchmark")
    parser.add_argument("--steps", type=int, default=200,
                        help="Training steps per config (default: 200)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset file (default: download names.txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: stdout)")
    parser.add_argument("--configs", type=str, default="v1,v2,v3",
                        help="Comma-separated configs to run (default: v1,v2,v3)")
    args = parser.parse_args()

    # Load data
    docs = load_dataset(args.dataset)
    tokenizer = Tokenizer.from_docs(docs)
    print(f"📊 Dataset: {len(docs)} docs, vocab_size={tokenizer.vocab_size}",
          file=sys.stderr)

    configs_to_run = args.configs.split(",")
    config_map = {
        "v1": ("v1-baseline", V1_CONFIG),
        "v2": ("v2-parameter-golf", V2_CONFIG),
        "v2_bigram": ("v2-parameter-golf+bigram", V2_BIGRAM_CONFIG),
        "v3": ("v3-parameter-golf", V3_CONFIG),
        "v3_smeargate": ("v3-parameter-golf+smeargate", V3_SMEARGATE_CONFIG),
    }

    results = []
    for name in configs_to_run:
        name = name.strip()
        if name in config_map:
            label, cfg = config_map[name]
            r = run_benchmark(cfg, docs, tokenizer, args.steps, args.lr, label)
            results.append(r)
        else:
            print(f"⚠️  Unknown config: {name}", file=sys.stderr)

    # Compare v1 vs v2
    comparison = {}
    v1_result = next((r for r in results if r["label"] == "v1-baseline"), None)
    v2_result = next((r for r in results if r["label"] == "v2-parameter-golf"), None)

    if v1_result and v2_result:
        loss_improvement = v1_result["losses"]["final"] - v2_result["losses"]["final"]
        loss_improvement_pct = (loss_improvement / v1_result["losses"]["final"]) * 100
        param_savings = v1_result["num_params"] - v2_result["num_params"]
        param_savings_pct = (param_savings / v1_result["num_params"]) * 100
        efficiency_improvement = (
            v1_result["efficiency"]["loss_x_params"]
            - v2_result["efficiency"]["loss_x_params"]
        )

        comparison = {
            "loss_improvement": round(loss_improvement, 6),
            "loss_improvement_pct": round(loss_improvement_pct, 2),
            "param_savings": param_savings,
            "param_savings_pct": round(param_savings_pct, 2),
            "efficiency_improvement": round(efficiency_improvement, 2),
            "v2_better_loss": v2_result["losses"]["final"] < v1_result["losses"]["final"],
            "v2_fewer_params": v2_result["num_params"] < v1_result["num_params"],
            "v2_better_efficiency": (
                v2_result["efficiency"]["loss_x_params"]
                < v1_result["efficiency"]["loss_x_params"]
            ),
            "v2_faster_convergence": (
                v2_result["losses"]["steps_to_3.0"]
                < v1_result["losses"]["steps_to_3.0"]
            ),
        }

    output = {
        "benchmark": "strands-microgpt",
        "version": MicroGPT.VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "names.txt" if not args.dataset else os.path.basename(args.dataset),
            "num_docs": len(docs),
            "vocab_size": tokenizer.vocab_size,
        },
        "hyperparameters": {
            "steps": args.steps,
            "learning_rate": args.lr,
            **SHARED,
        },
        "results": results,
        "comparison": comparison,
    }

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print("  BENCHMARK RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for r in results:
        print(f"  {r['label']:30s} | loss={r['losses']['final']:.4f} "
              f"| params={r['num_params']:,} "
              f"| efficiency={r['efficiency']['loss_x_params']:.0f}",
              file=sys.stderr)
    if comparison:
        print("\n  v2 vs v1:", file=sys.stderr)
        print(f"    Loss improvement:    {comparison['loss_improvement_pct']:+.1f}%", file=sys.stderr)
        print(f"    Param savings:       {comparison['param_savings_pct']:+.1f}%", file=sys.stderr)
        better = "✅" if comparison["v2_better_efficiency"] else "❌"
        print(f"    Better efficiency:   {better}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Output JSON
    json_str = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"\n📄 Results saved to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    # Exit code: 0 if v2 is better efficiency, 1 if regression
    if comparison and not comparison["v2_better_efficiency"]:
        print("\n❌ REGRESSION: v2 efficiency worse than v1!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
