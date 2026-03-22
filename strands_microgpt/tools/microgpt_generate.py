"""Generate text from a trained MicroGPT model as a Strands tool."""

import logging
import os
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)


@tool
def microgpt_generate(
    checkpoint_path: str = "/tmp/microgpt_checkpoint.json",
    num_samples: int = 20,
    temperature: float = 0.5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Generate text from a trained MicroGPT checkpoint.

    Loads a previously trained model and generates new samples.
    No training occurs — this is pure inference.
    Supports both v1 and v2 checkpoint formats.

    Args:
        checkpoint_path: Path to the model checkpoint.
        num_samples: Number of samples to generate.
        temperature: Sampling temperature (0, 1]. Lower = more conservative.
        seed: Random seed (0 = use default).

    Returns:
        Dict with generated samples.
    """
    try:
        import random

        from strands_microgpt.engine import MicroGPT

        if seed:
            random.seed(seed)

        if not os.path.exists(checkpoint_path):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Checkpoint not found: {checkpoint_path}\n"
                        f"Train first with microgpt_train()."
                    }
                ],
            }

        model, tokenizer, metadata = MicroGPT.load_checkpoint(checkpoint_path)

        samples = model.generate(
            tokenizer,
            num_samples=num_samples,
            temperature=temperature,
        )

        version = metadata.get("version", "v1")
        techniques = metadata.get("techniques", [])
        techniques_str = f"  Techniques: {', '.join(techniques)}\n" if techniques else ""

        result_text = (
            f"Generated {num_samples} samples (temperature={temperature}):\n"
            f"  Model ({version}): {model.num_params} params, "
            f"n_layer={model.n_layer}, n_embd={model.n_embd}, "
            f"{model.n_head}Q/{model.n_kv_head}KV\n"
            f"{techniques_str}\n"
        )
        for i, s in enumerate(samples, 1):
            result_text += f"  {i:2d}. {s}\n"

        return {
            "status": "success",
            "content": [{"text": result_text}],
        }

    except Exception as e:
        logger.error(f"microgpt_generate error: {e}")
        return {"status": "error", "content": [{"text": str(e)}]}
