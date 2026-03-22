"""Pure-Python GPT engine — autograd, transformer, tokenizer, training, inference.

This is Karpathy's "most atomic GPT" refactored into a reusable module.
Zero dependencies beyond Python stdlib. Everything is the algorithm.

Original: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
@karpathy — "The most atomic way to train and run inference for a GPT
in pure, dependency-free Python. This file is the complete algorithm.
Everything else is just efficiency."
"""

import json
import math
import os
import random
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Autograd Engine — recursive chain rule through a computation graph
# =============================================================================


class Value:
    """Scalar-valued autograd node. Tracks computation graph for backprop.

    Example:
        >>> a = Value(2.0)
        >>> b = Value(3.0)
        >>> c = a * b + a
        >>> c.backward()
        >>> a.grad  # dc/da = b + 1 = 4.0
        4.0
    """

    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        """Backpropagate gradients through the computation graph."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"


# =============================================================================
# Tokenizer — character-level, BOS-delimited
# =============================================================================


class Tokenizer:
    """Character-level tokenizer with BOS token.

    Example:
        >>> tok = Tokenizer.from_docs(["hello", "world"])
        >>> tok.encode("hello")
        [26, ...]
        >>> tok.decode([26, ...])
        'hello'
    """

    def __init__(self, chars: List[str]):
        self.chars = chars
        self.bos = len(chars)  # BOS token id
        self.vocab_size = len(chars) + 1
        self._char_to_id = {ch: i for i, ch in enumerate(chars)}

    @classmethod
    def from_docs(cls, docs: List[str]) -> "Tokenizer":
        """Build tokenizer from a list of documents."""
        chars = sorted(set("".join(docs)))
        return cls(chars)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tokenizer":
        """Reconstruct tokenizer from serialized dict."""
        return cls(data["chars"])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tokenizer."""
        return {"chars": self.chars}

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids (no BOS wrapping)."""
        return [self._char_to_id[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text (skips BOS)."""
        return "".join(self.chars[i] for i in ids if i != self.bos)


# =============================================================================
# MicroGPT — the transformer
# =============================================================================


def _matrix(nout: int, nin: int, std: float = 0.08) -> List[List[Value]]:
    """Initialize a weight matrix with random Gaussian values."""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def _linear(x: List[Value], w: List[List[Value]]) -> List[Value]:
    """Linear layer: y = Wx."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def _softmax(logits: List[Value]) -> List[Value]:
    """Numerically stable softmax."""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _rmsnorm(x: List[Value]) -> List[Value]:
    """Root mean square layer normalization."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


class MicroGPT:
    """Pure-Python GPT with autograd, attention, and Adam optimizer.

    Architecture follows GPT-2 with minor differences:
    - LayerNorm → RMSNorm
    - No biases
    - GeLU → ReLU

    Example:
        >>> gpt = MicroGPT(vocab_size=27, n_layer=1, n_embd=16, block_size=16, n_head=4)
        >>> gpt.train_on_docs(["hello", "world"], num_steps=100)
        >>> samples = gpt.generate(num_samples=5)
    """

    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 1,
        n_embd: int = 16,
        block_size: int = 16,
        n_head: int = 4,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.seed = seed

        random.seed(seed)

        # Initialize parameters
        self.state_dict: Dict[str, List[List[Value]]] = {
            "wte": _matrix(vocab_size, n_embd),
            "wpe": _matrix(block_size, n_embd),
            "lm_head": _matrix(vocab_size, n_embd),
        }
        for i in range(n_layer):
            self.state_dict[f"layer{i}.attn_wq"] = _matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wk"] = _matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wv"] = _matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wo"] = _matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.mlp_fc1"] = _matrix(4 * n_embd, n_embd)
            self.state_dict[f"layer{i}.mlp_fc2"] = _matrix(n_embd, 4 * n_embd)

        self.params = [
            p for mat in self.state_dict.values() for row in mat for p in row
        ]

    @property
    def num_params(self) -> int:
        return len(self.params)

    def forward(
        self,
        token_id: int,
        pos_id: int,
        keys: List[List],
        values: List[List],
    ) -> List[Value]:
        """Forward pass for a single token. Returns logits."""
        tok_emb = self.state_dict["wte"][token_id]
        pos_emb = self.state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = _rmsnorm(x)

        for li in range(self.n_layer):
            # Multi-head Attention
            x_residual = x
            x = _rmsnorm(x)
            q = _linear(x, self.state_dict[f"layer{li}.attn_wq"])
            k = _linear(x, self.state_dict[f"layer{li}.attn_wk"])
            v = _linear(x, self.state_dict[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)

            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs : hs + self.head_dim]
                k_h = [ki[hs : hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + self.head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim))
                    / self.head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = _softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

            x = _linear(x_attn, self.state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]

            # MLP
            x_residual = x
            x = _rmsnorm(x)
            x = _linear(x, self.state_dict[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = _linear(x, self.state_dict[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = _linear(x, self.state_dict["lm_head"])
        return logits

    def train_on_docs(
        self,
        docs: List[str],
        tokenizer: "Tokenizer",
        num_steps: int = 1000,
        learning_rate: float = 0.01,
        log_every: int = 100,
        callback=None,
    ) -> List[float]:
        """Train the model on a list of documents.

        Args:
            docs: List of text documents.
            tokenizer: Tokenizer instance.
            num_steps: Number of training steps.
            learning_rate: Initial learning rate.
            log_every: Log loss every N steps.
            callback: Optional callback(step, loss) for progress reporting.

        Returns:
            List of loss values at each step.
        """
        random.seed(self.seed)
        random.shuffle(docs)

        beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
        m = [0.0] * len(self.params)
        v = [0.0] * len(self.params)
        losses = []

        for step in range(num_steps):
            doc = docs[step % len(docs)]
            tokens = [tokenizer.bos] + tokenizer.encode(doc) + [tokenizer.bos]
            n = min(self.block_size, len(tokens) - 1)

            keys = [[] for _ in range(self.n_layer)]
            values = [[] for _ in range(self.n_layer)]
            step_losses = []

            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = self.forward(token_id, pos_id, keys, values)
                probs = _softmax(logits)
                loss_t = -probs[target_id].log()
                step_losses.append(loss_t)

            loss = (1 / n) * sum(step_losses)
            loss.backward()

            # Adam optimizer
            lr_t = learning_rate * (1 - step / num_steps)
            for i, p in enumerate(self.params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
                p.grad = 0

            losses.append(loss.data)

            if callback:
                callback(step, loss.data)

            if log_every and (step + 1) % log_every == 0:
                print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

        return losses

    def generate(
        self,
        tokenizer: "Tokenizer",
        num_samples: int = 20,
        temperature: float = 0.5,
        max_length: Optional[int] = None,
    ) -> List[str]:
        """Generate text samples from the trained model.

        Args:
            tokenizer: Tokenizer instance.
            num_samples: Number of samples to generate.
            temperature: Sampling temperature (0, 1]. Lower = more conservative.
            max_length: Maximum generation length (default: block_size).

        Returns:
            List of generated text strings.
        """
        max_len = max_length or self.block_size
        samples = []

        for _ in range(num_samples):
            keys = [[] for _ in range(self.n_layer)]
            values = [[] for _ in range(self.n_layer)]
            token_id = tokenizer.bos
            sample = []

            for pos_id in range(max_len):
                logits = self.forward(token_id, pos_id, keys, values)
                probs = _softmax([logit / temperature for logit in logits])
                token_id = random.choices(
                    range(tokenizer.vocab_size),
                    weights=[p.data for p in probs],
                )[0]
                if token_id == tokenizer.bos:
                    break
                sample.append(tokenizer.chars[token_id])

            samples.append("".join(sample))

        return samples

    def save_checkpoint(self, path: str, tokenizer: "Tokenizer", metadata: Optional[Dict] = None):
        """Save model weights and config to a JSON checkpoint.

        Args:
            path: File path to save checkpoint.
            tokenizer: Tokenizer to save alongside weights.
            metadata: Optional metadata dict.
        """
        checkpoint = {
            "config": {
                "vocab_size": self.vocab_size,
                "n_layer": self.n_layer,
                "n_embd": self.n_embd,
                "block_size": self.block_size,
                "n_head": self.n_head,
                "seed": self.seed,
            },
            "tokenizer": tokenizer.to_dict(),
            "weights": {
                name: [[v.data for v in row] for row in matrix]
                for name, matrix in self.state_dict.items()
            },
            "metadata": metadata or {},
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> Tuple["MicroGPT", "Tokenizer", Dict]:
        """Load model from a JSON checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Tuple of (MicroGPT model, Tokenizer, metadata dict).
        """
        with open(path, "r") as f:
            checkpoint = json.load(f)

        config = checkpoint["config"]
        model = cls(**config)
        tokenizer = Tokenizer.from_dict(checkpoint["tokenizer"])

        # Restore weights
        for name, weight_data in checkpoint["weights"].items():
            for i, row in enumerate(weight_data):
                for j, val in enumerate(row):
                    model.state_dict[name][i][j].data = val

        return model, tokenizer, checkpoint.get("metadata", {})

    @classmethod
    def from_dataset(
        cls,
        dataset_url: str = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
        dataset_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple["MicroGPT", "Tokenizer", List[str]]:
        """Convenience: load dataset, build tokenizer, create model.

        Args:
            dataset_url: URL to download dataset from.
            dataset_path: Local path override.
            **kwargs: Additional args for MicroGPT constructor.

        Returns:
            Tuple of (model, tokenizer, docs).
        """
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path) as f:
                docs = [line.strip() for line in f if line.strip()]
        else:
            local_path = dataset_path or "/tmp/microgpt_input.txt"
            if not os.path.exists(local_path):
                urllib.request.urlretrieve(dataset_url, local_path)
            with open(local_path) as f:
                docs = [line.strip() for line in f if line.strip()]

        tokenizer = Tokenizer.from_docs(docs)

        model_kwargs = {"vocab_size": tokenizer.vocab_size}
        model_kwargs.update(kwargs)
        model = cls(**model_kwargs)

        return model, tokenizer, docs
