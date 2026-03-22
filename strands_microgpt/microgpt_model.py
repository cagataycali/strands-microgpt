"""MicroGPT model provider for Strands Agents.

Uses Karpathy's pure-Python GPT for character-level text generation.
Zero external dependencies beyond strands-agents and Python stdlib.

This is primarily educational — it demonstrates the Strands Model interface
with the simplest possible transformer implementation.

Original: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import json
import logging
import os
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel
from strands.models.model import Model
from strands.types.content import ContentBlock, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from typing_extensions import TypedDict, Unpack, override

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
)


class MicroGPTModel(Model):
    """Karpathy's pure-Python GPT as a Strands Model provider.

    This model trains a tiny character-level GPT from scratch on any text dataset,
    then generates completions. No PyTorch, no CUDA, no dependencies.

    The model is trained on first use (or loaded from checkpoint) and generates
    text character by character using the learned patterns.

    Example:
        >>> from strands import Agent
        >>> from strands_microgpt import MicroGPTModel
        >>> model = MicroGPTModel()
        >>> agent = Agent(model=model)
        >>> agent("Generate some names")

    Example with custom dataset:
        >>> model = MicroGPTModel(
        ...     dataset_path="my_data.txt",
        ...     num_steps=2000,
        ...     n_layer=2,
        ...     n_embd=32,
        ... )
    """

    class MicroGPTConfig(TypedDict, total=False):
        """MicroGPT configuration."""

        dataset_url: str
        dataset_path: Optional[str]
        checkpoint_path: Optional[str]
        num_steps: int
        n_layer: int
        n_embd: int
        block_size: int
        n_head: int
        n_kv_head: Optional[int]
        tie_embeddings: bool
        use_rope: bool
        use_bigram_hash: bool
        learning_rate: float
        temperature: float
        num_samples: int
        seed: int
        params: Optional[Dict[str, Any]]

    def __init__(
        self,
        dataset_url: str = DEFAULT_DATASET_URL,
        **model_config: Unpack[MicroGPTConfig],
    ) -> None:
        """Initialize MicroGPT provider.

        Args:
            dataset_url: URL to download training dataset.
            **model_config: Configuration options.
        """
        self.config: Dict[str, Any] = {
            "dataset_url": dataset_url,
            "dataset_path": None,
            "checkpoint_path": None,
            "num_steps": 1000,
            "n_layer": 1,
            "n_embd": 16,
            "block_size": 16,
            "n_head": 4,
            "n_kv_head": None,
            "tie_embeddings": True,
            "use_rope": True,
            "use_bigram_hash": False,
            "learning_rate": 0.01,
            "temperature": 0.5,
            "num_samples": 20,
            "seed": 42,
            **model_config,
        }

        self._model = None
        self._tokenizer = None
        self._trained = False

        # Load from checkpoint if provided
        checkpoint = self.config.get("checkpoint_path")
        if checkpoint and os.path.exists(checkpoint):
            self._load_checkpoint(checkpoint)

    def _ensure_trained(self) -> None:
        """Ensure model is trained (lazy initialization)."""
        if self._trained:
            return

        from strands_microgpt.engine import MicroGPT

        checkpoint = self.config.get("checkpoint_path")
        if checkpoint and os.path.exists(checkpoint):
            self._load_checkpoint(checkpoint)
            return

        logger.info("Training MicroGPT from scratch...")
        model, tokenizer, docs = MicroGPT.from_dataset(
            dataset_url=self.config["dataset_url"],
            dataset_path=self.config.get("dataset_path"),
            n_layer=self.config["n_layer"],
            n_embd=self.config["n_embd"],
            block_size=self.config["block_size"],
            n_head=self.config["n_head"],
            n_kv_head=self.config.get("n_kv_head"),
            tie_embeddings=self.config.get("tie_embeddings", True),
            use_rope=self.config.get("use_rope", True),
            use_bigram_hash=self.config.get("use_bigram_hash", False),
            seed=self.config["seed"],
        )

        logger.info(
            f"Training: {model.num_params} params, {len(docs)} docs, "
            f"{self.config['num_steps']} steps"
        )
        model.train_on_docs(
            docs,
            tokenizer,
            num_steps=self.config["num_steps"],
            learning_rate=self.config["learning_rate"],
        )

        self._model = model
        self._tokenizer = tokenizer
        self._trained = True
        logger.info("MicroGPT training complete")

    def _load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        from strands_microgpt.engine import MicroGPT

        logger.info(f"Loading checkpoint: {path}")
        self._model, self._tokenizer, metadata = MicroGPT.load_checkpoint(path)
        self._trained = True
        logger.info(f"Checkpoint loaded: {metadata}")

    @override
    def update_config(self, **model_config: Unpack[MicroGPTConfig]) -> None:  # type: ignore[override]
        """Update configuration."""
        self.config.update(model_config)

    @override
    def get_config(self) -> MicroGPTConfig:
        """Get configuration."""
        return self.config  # type: ignore[return-value]

    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> Dict[str, Any]:
        """Format a content block."""
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        return {"type": "text", "text": "[unsupported content]"}

    @classmethod
    def format_request_message_tool_call(cls, tool_use: ToolUse) -> Dict[str, Any]:
        """Format a tool call."""
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @classmethod
    def format_request_tool_message(cls, tool_result: ToolResult) -> Dict[str, Any]:
        """Format a tool result message."""
        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )
        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": [
                cls.format_request_message_content(content) for content in contents
            ],
        }

    @classmethod
    def format_request_messages(
        cls, messages: Messages, system_prompt: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """Format messages array."""
        formatted = []
        if system_prompt:
            formatted.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        for message in messages:
            contents = message["content"]
            text_parts = []
            for content in contents:
                if "text" in content:
                    text_parts.append(content["text"])
                elif "toolResult" in content:
                    text_parts.append(str(content["toolResult"]))

            formatted.append({"role": message["role"], "content": " ".join(text_parts)})

        return formatted

    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a Strands request."""
        return {
            "messages": self.format_request_messages(messages, system_prompt),
            "tools": None,  # MicroGPT doesn't support tool calling
        }

    def format_chunk(self, event: Dict[str, Any]) -> StreamEvent:
        """Format an event into a StreamEvent."""
        chunk_type = event["chunk_type"]

        if chunk_type == "message_start":
            return {"messageStart": {"role": "assistant"}}
        if chunk_type == "content_start":
            return {"contentBlockStart": {"start": {}}}
        if chunk_type == "content_delta":
            return {"contentBlockDelta": {"delta": {"text": event["data"]}}}
        if chunk_type == "content_stop":
            return {"contentBlockStop": {}}
        if chunk_type == "message_stop":
            return {"messageStop": {"stopReason": "end_turn"}}
        if chunk_type == "metadata":
            return {
                "metadata": {
                    "usage": {
                        "inputTokens": event["data"].get("input_tokens", 0),
                        "outputTokens": event["data"].get("output_tokens", 0),
                        "totalTokens": event["data"].get("input_tokens", 0)
                        + event["data"].get("output_tokens", 0),
                    },
                    "metrics": {"latencyMs": 0},
                },
            }

        raise RuntimeError(f"chunk_type=<{chunk_type}> | unknown type")

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream text generation from MicroGPT.

        The model generates character-by-character samples based on
        its training data. Each sample is streamed as a chunk.

        Args:
            messages: List of message objects.
            tool_specs: Not supported (ignored).
            system_prompt: System prompt (used as context hint).
            tool_choice: Not supported (ignored).
            **kwargs: Additional arguments.

        Yields:
            Formatted message chunks.
        """
        self._ensure_trained()

        params = self.config.get("params", {})
        temperature = params.get("temperature", self.config.get("temperature", 0.5))
        num_samples = params.get("num_samples", self.config.get("num_samples", 20))

        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start"})

        # Generate samples
        samples = self._model.generate(
            self._tokenizer,
            num_samples=num_samples,
            temperature=temperature,
        )

        # Stream the output
        output_text = "Generated samples:\n"
        for i, sample in enumerate(samples, 1):
            output_text += f"  {i:2d}. {sample}\n"

        # Stream character by character for the streaming interface
        total_chars = 0
        for chunk in _chunk_text(output_text, chunk_size=10):
            yield self.format_chunk(
                {"chunk_type": "content_delta", "data": chunk}
            )
            total_chars += len(chunk)

        yield self.format_chunk({"chunk_type": "content_stop"})
        yield self.format_chunk({"chunk_type": "message_stop"})

        yield self.format_chunk(
            {
                "chunk_type": "metadata",
                "data": {"input_tokens": 0, "output_tokens": total_chars},
            }
        )

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Union[T, Any]], None]:
        """Structured output is not supported by MicroGPT."""
        raise NotImplementedError(
            "MicroGPT does not support structured output. "
            "Use a larger model for structured generation."
        )


def _chunk_text(text: str, chunk_size: int = 10) -> List[str]:
    """Split text into chunks for streaming."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
