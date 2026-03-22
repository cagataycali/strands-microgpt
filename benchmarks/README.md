# MicroGPT Benchmarks

Automated benchmarks comparing v1 (baseline) vs v2 (Parameter Golf) configurations.

## What's Measured

| Metric | Description | Better |
|--------|-------------|--------|
| **Final Loss** | Cross-entropy loss after N training steps | Lower ↓ |
| **BPB** | Bits per byte (loss / ln2) | Lower ↓ |
| **Params** | Total trainable parameters | Fewer ↓ |
| **Efficiency** | `loss × params` — the Parameter Golf metric | Lower ↓ |
| **Convergence** | Steps to reach loss < 3.0 | Fewer ↓ |
| **Unique Samples** | Diversity of generated text | Higher ↑ |

## Configurations

### v1-baseline
Standard MicroGPT: full KV heads, separate lm_head, learned position embeddings, no softcap.

### v2-parameter-golf
All winning techniques applied: GQA, tied embeddings, RoPE, ReLU², U-Net skips, softcap, gradient clipping, cosine warmdown.

### v2-parameter-golf+bigram
v2 + BigramHash embedding table from the #1 winner (thwu1).

## Running Locally

```bash
# Quick run (200 steps)
python benchmarks/benchmark.py

# Full benchmark (500 steps)
python benchmarks/benchmark.py --steps 500

# Save results
python benchmarks/benchmark.py --steps 500 --output results.json

# Only v1 vs v2
python benchmarks/benchmark.py --configs v1,v2
```

## CI Integration

The benchmark runs automatically on every PR via `.github/workflows/benchmark.yml`:

1. Trains v1 and v2 with identical hyperparameters
2. Compares loss, params, efficiency
3. Posts results as PR comment with comparison table
4. **Fails the build if v2 regresses** (efficiency worse than v1)

Historical results are stored in `gh-pages` branch artifacts.

## Interpreting Results

The key metric is **efficiency = loss × params**. This is what Parameter Golf optimizes:
train the best model with the fewest parameters.

A PR that reduces loss but adds too many params (or vice versa) will fail the benchmark.
