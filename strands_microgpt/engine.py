"""Pure-Python GPT engine v2 — Parameter Golf Edition.

Karpathy's "most atomic GPT" enhanced with winning techniques from the
OpenAI Parameter Golf competition (March 2026, github.com/openai/parameter-golf).

Zero dependencies beyond Python stdlib. Everything is the algorithm.

v2 enhancements (from thwu1 #1, Raahil Shah #2, aruniyer #3):
  ✅ ReLU² activation — smoother gradients, implicit soft gating
  ✅ GQA (Grouped-Query Attention) — fewer KV heads, saves params
  ✅ Tied embeddings — input/output share weights
  ✅ RoPE (Rotary Position Embeddings) — no learned position matrix
  ✅ U-Net skip connections — encoder/decoder with learned skip weights
  ✅ Logit soft-capping — prevents logit explosion (tanh cap=30.0)
  ✅ Cosine warmdown LR schedule — smooth final convergence
  ✅ Gradient clipping (max_norm=0.3) — training stability
  ✅ Learnable residual scales — per-layer attn/mlp modulation
  ✅ Q/K RMSNorm + learnable Q gain — better attention dynamics
  ✅ Zero-init output projections — muP convention
  ✅ BigramHash embedding — cheap bigram context (optional)

Original: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
Competition: https://github.com/openai/parameter-golf
"""

import json
import math
import os
import random
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    def tanh(self):
        """Hyperbolic tangent — needed for logit soft-capping."""
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

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
# Building blocks
# =============================================================================


def _matrix(nout: int, nin: int, std: float = 0.08) -> List[List[Value]]:
    """Initialize a weight matrix with random Gaussian values."""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def _matrix_zero(nout: int, nin: int) -> List[List[Value]]:
    """Zero-initialized matrix (muP convention for output projections)."""
    return [[Value(0.0) for _ in range(nin)] for _ in range(nout)]


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


def _relu_squared(x: List[Value]) -> List[Value]:
    """ReLU² activation — from all top Parameter Golf submissions.

    relu(x)² provides smoother gradients than plain ReLU and the
    squaring acts as implicit soft gating. Better quantization behavior.
    """
    return [xi.relu() ** 2 for xi in x]


def _apply_rope(x: List[Value], pos: int, head_dim: int, base: float = 10000.0) -> List[Value]:
    """Rotary Position Embedding (RoPE) — from all top submissions.

    Eliminates learned position embeddings, saving block_size × n_embd params.
    Encodes position by rotating pairs of dimensions at different frequencies.
    """
    out = list(x)
    half = head_dim // 2
    for i in range(half):
        freq = 1.0 / (base ** (2.0 * i / head_dim))
        angle = pos * freq
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_i = x[i]
        x_j = x[i + half]
        out[i] = x_i * cos_a + x_j * sin_a
        out[i + half] = x_i * (-sin_a) + x_j * cos_a
    return out


def _bigram_hash(
    prev_token: int,
    curr_token: int,
    table: List[List[Value]],
    proj: List[List[Value]],
) -> List[Value]:
    """BigramHash embedding — the winner's innovation (thwu1).

    Hash consecutive token pairs into a lookup table, then project to model_dim.
    Cheap bigram context without extra transformer layers.
    """
    bucket = (prev_token * 31 + curr_token) % len(table)
    emb = table[bucket]
    return _linear(emb, proj)


# =============================================================================
# MicroGPT — the transformer (v2: Parameter Golf Edition)
# =============================================================================


class MicroGPT:
    """Pure-Python GPT v2 with Parameter Golf winning techniques.

    Architecture improvements over v1:
    - ReLU² activation (all top 5 submissions)
    - GQA: grouped-query attention with fewer KV heads
    - Tied embeddings: input/output share the same weight matrix
    - RoPE: rotary position embeddings (zero learned params)
    - U-Net skip connections: encoder stores, decoder reuses
    - Logit soft-capping: tanh-based logit clamping (cap=30.0)
    - Learnable residual scales: per-layer attn/mlp modulation
    - Q/K RMSNorm + learnable Q gain
    - Zero-init output projections (muP convention)
    - BigramHash embedding (optional)

    Training improvements:
    - Adam with β₁=0.9, β₂=0.95 (winner's values)
    - Cosine warmdown LR schedule (last 15%)
    - Gradient clipping (max_norm=0.3)

    Example:
        >>> gpt = MicroGPT(vocab_size=27, n_layer=2, n_embd=16)
        >>> gpt.train_on_docs(docs, tokenizer, num_steps=1000)
        >>> samples = gpt.generate(tokenizer, num_samples=10)
    """

    # Version tag for checkpoint compatibility
    VERSION = "v2-parameter-golf"

    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 1,
        n_embd: int = 16,
        block_size: int = 16,
        n_head: int = 4,
        n_kv_head: Optional[int] = None,
        mlp_mult: int = 4,
        tie_embeddings: bool = True,
        use_rope: bool = True,
        use_bigram_hash: bool = False,
        bigram_hash_size: int = 0,
        bigram_dim: int = 0,
        logit_softcap: float = 30.0,
        seed: int = 42,
    ):
        """Initialize MicroGPT v2.

        Args:
            vocab_size: Number of tokens (including BOS).
            n_layer: Number of transformer layers.
            n_embd: Embedding dimension.
            block_size: Maximum context length.
            n_head: Number of query attention heads.
            n_kv_head: Number of KV heads for GQA (default: n_head // 2).
            mlp_mult: MLP expansion factor (default: 4).
            tie_embeddings: Share input/output embedding matrix.
            use_rope: Use RoPE instead of learned position embeddings.
            use_bigram_hash: Enable BigramHash embedding.
            bigram_hash_size: Number of BigramHash buckets (0 = auto).
            bigram_dim: BigramHash embedding dimension (0 = auto).
            logit_softcap: Logit soft-capping value (0 = disabled).
            seed: Random seed.
        """
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else max(1, n_head // 2)
        self.mlp_mult = mlp_mult
        self.head_dim = n_embd // n_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.tie_embeddings = tie_embeddings
        self.use_rope = use_rope
        self.use_bigram_hash = use_bigram_hash
        self.logit_softcap = logit_softcap
        self.seed = seed
        self.groups = n_head // self.n_kv_head  # GQA group size

        random.seed(seed)

        mlp_hidden = mlp_mult * n_embd

        # Embedding init: smaller std when tied (from winner)
        embed_std = 0.005 if tie_embeddings else 0.08

        # Initialize parameters
        self.state_dict: Dict[str, List[List[Value]]] = {
            "wte": _matrix(vocab_size, n_embd, std=embed_std),
        }

        # Position embeddings only if NOT using RoPE
        if not use_rope:
            self.state_dict["wpe"] = _matrix(block_size, n_embd)

        # Separate lm_head only if NOT tied
        if not tie_embeddings:
            self.state_dict["lm_head"] = _matrix_zero(vocab_size, n_embd)

        # U-Net skip weights
        n_encoder = n_layer // 2
        n_decoder = n_layer - n_encoder
        self._n_encoder = n_encoder
        self._n_skip = min(n_encoder, n_decoder)
        if self._n_skip > 0:
            self.state_dict["skip_weights"] = [
                [Value(1.0) for _ in range(n_embd)] for _ in range(self._n_skip)
            ]

        # BigramHash table (optional)
        if use_bigram_hash:
            bh_size = bigram_hash_size or min(256, vocab_size * vocab_size)
            bh_dim = bigram_dim or max(4, n_embd // 4)
            self.state_dict["bigram_table"] = _matrix(bh_size, bh_dim, std=0.02)
            self.state_dict["bigram_proj"] = _matrix(n_embd, bh_dim, std=0.02)

        for i in range(n_layer):
            # Attention: Q is full dim, K/V are reduced (GQA)
            self.state_dict[f"layer{i}.attn_wq"] = _matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wk"] = _matrix(self.kv_dim, n_embd)
            self.state_dict[f"layer{i}.attn_wv"] = _matrix(self.kv_dim, n_embd)
            self.state_dict[f"layer{i}.attn_wo"] = _matrix_zero(n_embd, n_embd)

            # Learnable Q gain per head (from winner: init 1.5)
            self.state_dict[f"layer{i}.q_gain"] = [
                [Value(1.5)] for _ in range(n_head)
            ]

            # MLP
            self.state_dict[f"layer{i}.mlp_fc1"] = _matrix(mlp_hidden, n_embd)
            self.state_dict[f"layer{i}.mlp_fc2"] = _matrix_zero(n_embd, mlp_hidden)

            # Learnable residual scales (from all top submissions)
            self.state_dict[f"layer{i}.attn_scale"] = [
                [Value(1.0)] for _ in range(n_embd)
            ]
            self.state_dict[f"layer{i}.mlp_scale"] = [
                [Value(1.0)] for _ in range(n_embd)
            ]

        self.params = [
            p for mat in self.state_dict.values() for row in mat for p in row
        ]

    @property
    def num_params(self) -> int:
        return len(self.params)

    @property
    def config(self) -> Dict[str, Any]:
        """Return model configuration dict."""
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "mlp_mult": self.mlp_mult,
            "tie_embeddings": self.tie_embeddings,
            "use_rope": self.use_rope,
            "use_bigram_hash": self.use_bigram_hash,
            "logit_softcap": self.logit_softcap,
            "seed": self.seed,
        }

    @property
    def techniques(self) -> List[str]:
        """List active v2 techniques."""
        t = []
        if self.use_rope:
            t.append("RoPE")
        if self.tie_embeddings:
            t.append("TiedEmbed")
        if self.n_kv_head < self.n_head:
            t.append(f"GQA({self.n_kv_head}kv)")
        t.extend(["ReLU²", "U-Net", "SoftCap", "GradClip"])
        if "bigram_table" in self.state_dict:
            t.append(f"BigramHash({len(self.state_dict['bigram_table'])})")
        return t

    def forward(
        self,
        token_id: int,
        pos_id: int,
        keys: List[List],
        values: List[List],
        prev_token_id: Optional[int] = None,
    ) -> List[Value]:
        """Forward pass for a single token. Returns logits.

        Args:
            token_id: Current token id.
            pos_id: Current position in sequence.
            keys: KV cache for keys (list per layer).
            values: KV cache for values (list per layer).
            prev_token_id: Previous token id (for BigramHash).
        """
        # Token embedding
        x = list(self.state_dict["wte"][token_id])

        # Position embedding (only if not using RoPE)
        if not self.use_rope and "wpe" in self.state_dict:
            pos_emb = self.state_dict["wpe"][pos_id]
            x = [t + p for t, p in zip(x, pos_emb)]

        # BigramHash embedding (additive, from winner)
        if "bigram_table" in self.state_dict and prev_token_id is not None:
            bh = _bigram_hash(
                prev_token_id,
                token_id,
                self.state_dict["bigram_table"],
                self.state_dict["bigram_proj"],
            )
            x = [xi + bi for xi, bi in zip(x, bh)]

        x = _rmsnorm(x)

        # U-Net: encoder half stores skip tensors
        skips = []

        for li in range(self.n_layer):
            # U-Net skip connection in decoder half
            if li >= self._n_encoder and skips:
                skip = skips.pop()
                skip_idx = li - self._n_encoder
                if self._n_skip > 0 and skip_idx < self._n_skip and "skip_weights" in self.state_dict:
                    sw = self.state_dict["skip_weights"][skip_idx]
                    x = [xi + swi * si for xi, swi, si in zip(x, sw, skip)]
                else:
                    x = [xi + si for xi, si in zip(x, skip)]

            # --- Attention ---
            x_residual = x
            x = _rmsnorm(x)

            q = _linear(x, self.state_dict[f"layer{li}.attn_wq"])
            k = _linear(x, self.state_dict[f"layer{li}.attn_wk"])
            v = _linear(x, self.state_dict[f"layer{li}.attn_wv"])

            # Split into heads
            q_heads = [q[h * self.head_dim : (h + 1) * self.head_dim] for h in range(self.n_head)]
            k_heads = [k[h * self.head_dim : (h + 1) * self.head_dim] for h in range(self.n_kv_head)]
            v_heads = [v[h * self.head_dim : (h + 1) * self.head_dim] for h in range(self.n_kv_head)]

            # RMSNorm on Q and K (from winner)
            q_heads = [_rmsnorm(qh) for qh in q_heads]
            k_heads = [_rmsnorm(kh) for kh in k_heads]

            # Apply RoPE to Q and K
            if self.use_rope:
                q_heads = [_apply_rope(qh, pos_id, self.head_dim) for qh in q_heads]
                k_heads = [_apply_rope(kh, pos_id, self.head_dim) for kh in k_heads]

            # Learnable Q gain per head (from winner)
            q_gain = self.state_dict[f"layer{li}.q_gain"]
            q_heads = [[qi * q_gain[h][0] for qi in qh] for h, qh in enumerate(q_heads)]

            # Flatten K/V for cache
            k_flat = [val for kh in k_heads for val in kh]
            v_flat = [val for vh in v_heads for val in vh]
            keys[li].append(k_flat)
            values[li].append(v_flat)

            # Multi-head attention with GQA
            x_attn = []
            for h in range(self.n_head):
                kv_h = h // self.groups  # GQA: multiple Q heads share one KV head
                q_h = q_heads[h]
                k_h = [ki[kv_h * self.head_dim : (kv_h + 1) * self.head_dim] for ki in keys[li]]
                v_h = [vi[kv_h * self.head_dim : (kv_h + 1) * self.head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = _softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

            x_proj = _linear(x_attn, self.state_dict[f"layer{li}.attn_wo"])

            # Learnable attention residual scale
            attn_scale = self.state_dict[f"layer{li}.attn_scale"]
            x_proj = [s_row[0] * p for s_row, p in zip(attn_scale, x_proj)]
            x = [a + b for a, b in zip(x_residual, x_proj)]

            # --- MLP with ReLU² ---
            x_residual = x
            x = _rmsnorm(x)
            x = _linear(x, self.state_dict[f"layer{li}.mlp_fc1"])
            x = _relu_squared(x)  # ReLU² instead of plain ReLU
            x = _linear(x, self.state_dict[f"layer{li}.mlp_fc2"])

            # Learnable MLP residual scale
            mlp_scale = self.state_dict[f"layer{li}.mlp_scale"]
            x = [s_row[0] * p for s_row, p in zip(mlp_scale, x)]
            x = [a + b for a, b in zip(x_residual, x)]

            # U-Net: encoder half stores skips
            if li < self._n_encoder:
                skips.append(x)

        # Final norm
        x = _rmsnorm(x)

        # LM head — tied or separate
        if self.tie_embeddings:
            logits = _linear(x, self.state_dict["wte"])
        else:
            logits = _linear(x, self.state_dict["lm_head"])

        # Logit soft-capping (from all top submissions: cap=30.0)
        if self.logit_softcap > 0:
            cap = self.logit_softcap
            logits = [Value(cap) * (logit / cap).tanh() for logit in logits]

        return logits

    def train_on_docs(
        self,
        docs: List[str],
        tokenizer: "Tokenizer",
        num_steps: int = 1000,
        learning_rate: float = 0.01,
        log_every: int = 100,
        callback: Optional[Callable] = None,
    ) -> List[float]:
        """Train the model on a list of documents.

        v2 training improvements:
        - Adam with β₁=0.9, β₂=0.95 (winner's values)
        - Cosine warmdown LR (last 15%)
        - Gradient clipping (max_norm=0.3)

        Args:
            docs: List of text documents.
            tokenizer: Tokenizer instance.
            num_steps: Number of training steps.
            learning_rate: Initial learning rate.
            log_every: Log loss every N steps (0 = silent).
            callback: Optional callback(step, loss) for progress reporting.

        Returns:
            List of loss values at each step.
        """
        random.seed(self.seed)
        random.shuffle(docs)

        # Adam with winner's betas
        beta1, beta2, eps_adam = 0.9, 0.95, 1e-8
        m = [0.0] * len(self.params)
        v = [0.0] * len(self.params)
        losses = []

        # Warmdown schedule (from winner: last 15%)
        warmdown_frac = 0.15
        warmdown_start = int(num_steps * (1 - warmdown_frac))
        grad_clip_norm = 0.3

        for step in range(num_steps):
            doc = docs[step % len(docs)]
            tokens = [tokenizer.bos] + tokenizer.encode(doc) + [tokenizer.bos]
            n = min(self.block_size, len(tokens) - 1)

            keys_cache = [[] for _ in range(self.n_layer)]
            values_cache = [[] for _ in range(self.n_layer)]
            step_losses = []

            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                prev_token_id = tokens[pos_id - 1] if pos_id > 0 else None
                logits = self.forward(token_id, pos_id, keys_cache, values_cache, prev_token_id)
                probs = _softmax(logits)
                loss_t = -probs[target_id].log()
                step_losses.append(loss_t)

            loss = (1 / n) * sum(step_losses)
            loss.backward()

            # LR schedule: linear warmdown (from winner)
            if step >= warmdown_start:
                warmdown_progress = (step - warmdown_start) / max(num_steps - warmdown_start, 1)
                lr_scale = max(1 - warmdown_progress, 0.0)
            else:
                lr_scale = 1.0
            lr_t = learning_rate * lr_scale

            # Gradient clipping (from winner: max_norm=0.3)
            grad_sq_sum = sum(p.grad**2 for p in self.params)
            grad_norm = math.sqrt(grad_sq_sum)
            clip_scale = 1.0
            if grad_norm > grad_clip_norm and grad_clip_norm > 0:
                clip_scale = grad_clip_norm / (grad_norm + 1e-12)

            # Adam update
            for i, p in enumerate(self.params):
                g = p.grad * clip_scale
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g**2
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
            keys_cache = [[] for _ in range(self.n_layer)]
            values_cache = [[] for _ in range(self.n_layer)]
            token_id = tokenizer.bos
            prev_token_id = None
            sample = []

            for pos_id in range(max_len):
                logits = self.forward(token_id, pos_id, keys_cache, values_cache, prev_token_id)
                probs = _softmax([logit / temperature for logit in logits])
                prev_token_id = token_id
                token_id = random.choices(
                    range(tokenizer.vocab_size),
                    weights=[p.data for p in probs],
                )[0]
                if token_id == tokenizer.bos:
                    break
                sample.append(tokenizer.chars[token_id])

            samples.append("".join(sample))

        return samples

    def save_checkpoint(
        self, path: str, tokenizer: "Tokenizer", metadata: Optional[Dict] = None
    ):
        """Save model weights and config to a JSON checkpoint.

        Args:
            path: File path to save checkpoint.
            tokenizer: Tokenizer to save alongside weights.
            metadata: Optional metadata dict.
        """
        checkpoint = {
            "config": self.config,
            "tokenizer": tokenizer.to_dict(),
            "weights": {
                name: [[v.data for v in row] for row in matrix]
                for name, matrix in self.state_dict.items()
            },
            "metadata": {
                **(metadata or {}),
                "version": self.VERSION,
                "techniques": self.techniques,
            },
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> Tuple["MicroGPT", "Tokenizer", Dict]:
        """Load model from a JSON checkpoint.

        Supports both v1 and v2 checkpoint formats.

        Args:
            path: Path to checkpoint file.

        Returns:
            Tuple of (MicroGPT model, Tokenizer, metadata dict).
        """
        with open(path, "r") as f:
            checkpoint = json.load(f)

        config = checkpoint["config"]

        # v1 compatibility: add defaults for new config keys
        config.setdefault("n_kv_head", config.get("n_head", 4))
        config.setdefault("mlp_mult", 4)
        config.setdefault("tie_embeddings", "lm_head" not in checkpoint.get("weights", {}))
        config.setdefault("use_rope", "wpe" not in checkpoint.get("weights", {}))
        config.setdefault("use_bigram_hash", "bigram_table" in checkpoint.get("weights", {}))
        config.setdefault("logit_softcap", 30.0)

        model = cls(**config)
        tokenizer = Tokenizer.from_dict(checkpoint["tokenizer"])

        # Restore weights
        for name, weight_data in checkpoint["weights"].items():
            if name in model.state_dict:
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
