"""Test v2 Parameter Golf enhancements to the engine."""

import math

from strands_microgpt.engine import (
    MicroGPT,
    Tokenizer,
    Value,
    _apply_rope,
    _bigram_hash,
    _matrix,
    _relu_squared,
    _rmsnorm,
)


# =============================================================================
# Autograd: tanh
# =============================================================================


def test_value_tanh():
    """tanh() returns correct value and gradient."""
    v = Value(2.0)
    t = v.tanh()
    assert abs(t.data - math.tanh(2.0)) < 1e-6
    t.backward()
    expected_grad = 1 - math.tanh(2.0) ** 2
    assert abs(v.grad - expected_grad) < 1e-6


def test_value_tanh_zero():
    """tanh(0) = 0, grad = 1."""
    v = Value(0.0)
    t = v.tanh()
    assert abs(t.data) < 1e-6
    t.backward()
    assert abs(v.grad - 1.0) < 1e-6


# =============================================================================
# ReLU²
# =============================================================================


def test_relu_squared_positive():
    x = [Value(2.0)]
    result = _relu_squared(x)
    assert abs(result[0].data - 4.0) < 1e-6  # relu(2)² = 4


def test_relu_squared_negative():
    x = [Value(-1.0)]
    result = _relu_squared(x)
    assert abs(result[0].data) < 1e-6  # relu(-1)² = 0


def test_relu_squared_fractional():
    x = [Value(0.5)]
    result = _relu_squared(x)
    assert abs(result[0].data - 0.25) < 1e-6  # relu(0.5)² = 0.25


def test_relu_squared_backward():
    """Gradient of relu²(x) = 2*relu(x) for x > 0."""
    x = [Value(3.0)]
    result = _relu_squared(x)
    result[0].backward()
    # d/dx relu(x)² = 2*relu(x) * d_relu(x) = 2*3*1 = 6
    assert abs(x[0].grad - 6.0) < 1e-6


# =============================================================================
# RoPE
# =============================================================================


def test_rope_identity_at_pos_zero():
    """At position 0, all angles are 0, so RoPE is identity."""
    x = [Value(1.0), Value(2.0), Value(3.0), Value(4.0)]
    result = _apply_rope(x, pos=0, head_dim=4)
    for i in range(4):
        assert abs(result[i].data - x[i].data) < 1e-6


def test_rope_rotates_at_nonzero_pos():
    """At position > 0, values should be rotated."""
    x = [Value(1.0), Value(0.0), Value(0.0), Value(0.0)]
    result = _apply_rope(x, pos=1, head_dim=4)
    # First pair should be rotated by freq-dependent angle
    assert abs(result[0].data - 1.0) > 1e-6 or abs(result[2].data) > 1e-6


def test_rope_preserves_norm():
    """RoPE is a rotation, so it should preserve vector norm."""
    x = [Value(1.0), Value(2.0), Value(3.0), Value(4.0)]
    result = _apply_rope(x, pos=5, head_dim=4)
    norm_before = sum(v.data**2 for v in x) ** 0.5
    norm_after = sum(v.data**2 for v in result) ** 0.5
    assert abs(norm_before - norm_after) < 1e-5


# =============================================================================
# GQA
# =============================================================================


def test_gqa_kv_dims():
    """GQA should have smaller K/V dimensions than Q."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=8, block_size=4, n_head=4)
    assert model.n_kv_head == 2  # default: n_head // 2
    assert model.kv_dim == 4  # 2 * (8 // 4) = 4
    # K/V weight matrices should be kv_dim × n_embd
    assert len(model.state_dict["layer0.attn_wk"]) == 4  # kv_dim
    assert len(model.state_dict["layer0.attn_wv"]) == 4
    # Q should be full n_embd
    assert len(model.state_dict["layer0.attn_wq"]) == 8


def test_gqa_explicit_kv_heads():
    """Explicit n_kv_head setting."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=8, block_size=4, n_head=4, n_kv_head=1)
    assert model.n_kv_head == 1
    assert len(model.state_dict["layer0.attn_wk"]) == 2  # 1 * head_dim=2


# =============================================================================
# Tied Embeddings
# =============================================================================


def test_tied_embeddings_no_lm_head():
    """With tied embeddings, there should be no separate lm_head."""
    model = MicroGPT(vocab_size=10, n_layer=1, n_embd=8, block_size=4, n_head=2, tie_embeddings=True)
    assert "lm_head" not in model.state_dict
    assert "wte" in model.state_dict


def test_untied_embeddings_has_lm_head():
    """Without tied embeddings, lm_head should exist."""
    model = MicroGPT(vocab_size=10, n_layer=1, n_embd=8, block_size=4, n_head=2, tie_embeddings=False)
    assert "lm_head" in model.state_dict


def test_tied_embeddings_fewer_params():
    """Tied embeddings should have fewer parameters."""
    tied = MicroGPT(vocab_size=10, n_layer=1, n_embd=8, block_size=4, n_head=2, tie_embeddings=True)
    untied = MicroGPT(vocab_size=10, n_layer=1, n_embd=8, block_size=4, n_head=2, tie_embeddings=False)
    assert tied.num_params < untied.num_params
    # Difference should be vocab_size * n_embd = 80
    assert untied.num_params - tied.num_params == 10 * 8


# =============================================================================
# RoPE vs learned position embeddings
# =============================================================================


def test_rope_no_wpe():
    """With RoPE, there should be no learned position embeddings."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=8, block_size=4, n_head=2, use_rope=True)
    assert "wpe" not in model.state_dict


def test_no_rope_has_wpe():
    """Without RoPE, learned position embeddings should exist."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=8, block_size=4, n_head=2, use_rope=False)
    assert "wpe" in model.state_dict
    assert len(model.state_dict["wpe"]) == 4  # block_size


# =============================================================================
# U-Net Skip Connections
# =============================================================================


def test_skip_connections_exist():
    """Multi-layer models should have skip_weights."""
    model = MicroGPT(vocab_size=5, n_layer=4, n_embd=8, block_size=4, n_head=2)
    assert "skip_weights" in model.state_dict
    # 4 layers: 2 encoder, 2 decoder, 2 skips
    assert len(model.state_dict["skip_weights"]) == 2


def test_single_layer_no_skips():
    """Single-layer model should not have skip_weights."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=8, block_size=4, n_head=2)
    # n_encoder = 0, n_skip = 0, so no skip_weights
    assert "skip_weights" not in model.state_dict


# =============================================================================
# Logit Soft-Capping
# =============================================================================


def test_softcap_clamps_logits():
    """Soft-capped logits should be bounded by ±cap."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=4, block_size=4, n_head=2, logit_softcap=30.0)
    keys = [[] for _ in range(1)]
    values = [[] for _ in range(1)]
    logits = model.forward(0, 0, keys, values)
    for l in logits:
        assert abs(l.data) <= 30.0 + 0.01  # bounded by cap


def test_softcap_disabled():
    """logit_softcap=0 should disable capping."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=4, block_size=4, n_head=2, logit_softcap=0)
    keys = [[] for _ in range(1)]
    values = [[] for _ in range(1)]
    logits = model.forward(0, 0, keys, values)
    # Should still produce logits (no capping applied)
    assert len(logits) == 5


# =============================================================================
# Learnable residual scales and Q gain
# =============================================================================


def test_residual_scales_exist():
    model = MicroGPT(vocab_size=5, n_layer=2, n_embd=8, block_size=4, n_head=2)
    assert "layer0.attn_scale" in model.state_dict
    assert "layer0.mlp_scale" in model.state_dict
    assert "layer1.attn_scale" in model.state_dict
    # Check dimensions
    assert len(model.state_dict["layer0.attn_scale"]) == 8  # n_embd


def test_q_gain_exists():
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=8, block_size=4, n_head=4)
    assert "layer0.q_gain" in model.state_dict
    assert len(model.state_dict["layer0.q_gain"]) == 4  # n_head


# =============================================================================
# BigramHash
# =============================================================================


def test_bigram_hash_optional():
    """BigramHash should be off by default."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=8, block_size=4, n_head=2)
    assert "bigram_table" not in model.state_dict


def test_bigram_hash_enabled():
    """BigramHash creates table and projection."""
    model = MicroGPT(
        vocab_size=5, n_layer=1, n_embd=8, block_size=4, n_head=2,
        use_bigram_hash=True,
    )
    assert "bigram_table" in model.state_dict
    assert "bigram_proj" in model.state_dict


# =============================================================================
# Backward through all new ops
# =============================================================================


def test_backward_through_v2_model():
    """Full backward pass through v2 model with all features."""
    from strands_microgpt.engine import _softmax

    model = MicroGPT(vocab_size=5, n_layer=2, n_embd=4, block_size=4, n_head=2)
    keys = [[] for _ in range(2)]
    values = [[] for _ in range(2)]

    logits = model.forward(0, 0, keys, values)
    probs = _softmax(logits)
    loss = -probs[1].log()
    loss.backward()

    # Embedding always gets gradients (it's the input AND output with tied weights)
    assert model.state_dict["wte"][0][0].grad != 0

    # Count total nonzero gradients — should be substantial
    nonzero_grads = sum(
        1 for p in model.params if p.grad != 0
    )
    assert nonzero_grads > 0, "At least some params must have nonzero gradients"


# =============================================================================
# End-to-end: train + generate with v2
# =============================================================================


def test_train_and_generate_v2():
    """End-to-end training and generation with all v2 features."""
    docs = ["abc", "bca", "cab"] * 20
    tok = Tokenizer.from_docs(docs)
    model = MicroGPT(
        vocab_size=tok.vocab_size,
        n_layer=2,
        n_embd=8,
        block_size=8,
        n_head=2,
        seed=42,
    )

    losses = model.train_on_docs(docs, tok, num_steps=50, learning_rate=0.01, log_every=0)
    assert len(losses) == 50
    assert losses[-1] < losses[0]  # Loss should decrease

    samples = model.generate(tok, num_samples=5, temperature=0.8)
    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, str)


def test_train_generate_with_bigram_hash():
    """Train and generate with BigramHash enabled."""
    docs = ["abc", "bca", "cab"] * 20
    tok = Tokenizer.from_docs(docs)
    model = MicroGPT(
        vocab_size=tok.vocab_size,
        n_layer=2,
        n_embd=8,
        block_size=8,
        n_head=2,
        use_bigram_hash=True,
        seed=42,
    )

    losses = model.train_on_docs(docs, tok, num_steps=30, learning_rate=0.01, log_every=0)
    assert len(losses) == 30
    samples = model.generate(tok, num_samples=3)
    assert len(samples) == 3


# =============================================================================
# Checkpoint v2 roundtrip
# =============================================================================


def test_checkpoint_v2_roundtrip(tmp_path):
    """Save and load a v2 checkpoint with all features."""
    docs = ["hello", "world"] * 20
    tok = Tokenizer.from_docs(docs)
    model = MicroGPT(
        vocab_size=tok.vocab_size,
        n_layer=2,
        n_embd=8,
        block_size=8,
        n_head=2,
        seed=42,
    )
    model.train_on_docs(docs, tok, num_steps=10, log_every=0)

    path = str(tmp_path / "v2.json")
    model.save_checkpoint(path, tok, metadata={"test": True})

    model2, tok2, meta = MicroGPT.load_checkpoint(path)
    assert model2.num_params == model.num_params
    assert model2.n_kv_head == model.n_kv_head
    assert model2.tie_embeddings == model.tie_embeddings
    assert model2.use_rope == model.use_rope
    assert meta["version"] == "v2-parameter-golf"
    assert meta["test"] is True
    assert "RoPE" in meta["techniques"]

    samples = model2.generate(tok2, num_samples=3)
    assert len(samples) == 3


# =============================================================================
# Techniques list
# =============================================================================


def test_techniques_list():
    model = MicroGPT(vocab_size=5, n_layer=2, n_embd=8, block_size=4, n_head=4)
    t = model.techniques
    assert "RoPE" in t
    assert "TiedEmbed" in t
    assert "GQA(2kv)" in t
    assert "ReLU²" in t
    assert "U-Net" in t
    assert "SoftCap" in t
    assert "GradClip" in t


# =============================================================================
# Config property
# =============================================================================


def test_config_property():
    model = MicroGPT(vocab_size=5, n_layer=2, n_embd=8, block_size=4, n_head=4)
    c = model.config
    assert c["n_layer"] == 2
    assert c["n_kv_head"] == 2
    assert c["tie_embeddings"] is True
    assert c["use_rope"] is True
    assert c["logit_softcap"] == 30.0
