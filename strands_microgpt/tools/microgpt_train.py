"""Train a MicroGPT model as a Strands tool."""

import logging
import time
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)

_cached_model = None
_cached_tokenizer = None


@tool
def microgpt_train(
    dataset_url: str = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
    dataset_path: str = "",
    num_steps: int = 1000,
    n_layer: int = 1,
    n_embd: int = 16,
    block_size: int = 16,
    n_head: int = 4,
    learning_rate: float = 0.01,
    seed: int = 42,
    checkpoint_path: str = "",
    tie_embeddings: bool = True,
    use_rope: bool = True,
    use_bigram_hash: bool = False,
) -> Dict[str, Any]:
    """Train Karpathy's pure-Python GPT from scratch on a text dataset.

    Zero dependencies. Pure autograd. The complete algorithm in Python.

    v2 enhancements from OpenAI Parameter Golf winners:
    - ReLU² activation (smoother gradients)
    - GQA (Grouped-Query Attention)
    - Tied embeddings (input/output share weights)
    - RoPE (Rotary Position Embeddings)
    - U-Net skip connections
    - Logit soft-capping (cap=30.0)
    - Cosine warmdown LR + gradient clipping
    - Learnable residual scales

    Args:
        dataset_url: URL to download dataset (default: names.txt).
        dataset_path: Local path to dataset file (overrides URL).
        num_steps: Number of training steps.
        n_layer: Number of transformer layers.
        n_embd: Embedding dimension.
        block_size: Maximum context length.
        n_head: Number of attention heads.
        learning_rate: Initial learning rate.
        seed: Random seed.
        checkpoint_path: Path to save checkpoint after training.
        tie_embeddings: Share input/output embedding matrix.
        use_rope: Use RoPE instead of learned position embeddings.
        use_bigram_hash: Enable BigramHash embedding (winner's technique).

    Returns:
        Dict with training stats, loss history, and sample generations.
    """
    global _cached_model, _cached_tokenizer

    try:
        from strands_microgpt.engine import MicroGPT

        t0 = time.time()

        # Build model + tokenizer + load dataset
        model, tokenizer, docs = MicroGPT.from_dataset(
            dataset_url=dataset_url,
            dataset_path=dataset_path if dataset_path else None,
            n_layer=n_layer,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            seed=seed,
            tie_embeddings=tie_embeddings,
            use_rope=use_rope,
            use_bigram_hash=use_bigram_hash,
        )

        info = (
            f"Training MicroGPT v2: {model.num_params} params, "
            f"{len(docs)} docs, {num_steps} steps, "
            f"n_layer={n_layer}, n_embd={n_embd}, "
            f"n_head={n_head}Q/{model.n_kv_head}KV"
        )
        logger.info(info)

        # Train
        losses = model.train_on_docs(
            docs,
            tokenizer,
            num_steps=num_steps,
            learning_rate=learning_rate,
            log_every=max(1, num_steps // 10),
        )

        train_time = time.time() - t0

        # Generate samples
        samples = model.generate(tokenizer, num_samples=20, temperature=0.5)

        # Cache for later generation
        _cached_model = model
        _cached_tokenizer = tokenizer

        # Save checkpoint if path provided
        save_path = checkpoint_path or "/tmp/microgpt_checkpoint.json"
        model.save_checkpoint(
            save_path,
            tokenizer,
            metadata={
                "num_steps": num_steps,
                "final_loss": losses[-1] if losses else None,
                "train_time_seconds": train_time,
            },
        )

        techniques = ", ".join(model.techniques)

        result_text = (
            f"✅ MicroGPT v2 trained successfully!\n"
            f"  Params: {model.num_params}\n"
            f"  Heads: {model.n_head}Q / {model.n_kv_head}KV (GQA)\n"
            f"  Techniques: {techniques}\n"
            f"  Dataset: {len(docs)} documents\n"
            f"  Steps: {num_steps}\n"
            f"  Final loss: {losses[-1]:.4f}\n"
            f"  Time: {train_time:.1f}s\n"
            f"  Checkpoint: {save_path}\n\n"
            f"Generated samples:\n"
        )
        for i, s in enumerate(samples, 1):
            result_text += f"  {i:2d}. {s}\n"

        return {
            "status": "success",
            "content": [{"text": result_text}],
        }

    except Exception as e:
        logger.error(f"microgpt_train error: {e}")
        return {"status": "error", "content": [{"text": str(e)}]}
