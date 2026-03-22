"""Test tokenizer and model training/generation."""

from strands_microgpt.engine import MicroGPT, Tokenizer


def test_tokenizer_roundtrip():
    docs = ["hello", "world", "test"]
    tok = Tokenizer.from_docs(docs)

    for doc in docs:
        encoded = tok.encode(doc)
        decoded = tok.decode(encoded)
        assert decoded == doc


def test_tokenizer_vocab_size():
    docs = ["abc", "bcd"]
    tok = Tokenizer.from_docs(docs)
    # Unique chars: a, b, c, d = 4 + 1 BOS = 5
    assert tok.vocab_size == 5
    assert tok.bos == 4


def test_tokenizer_serialization():
    docs = ["hello", "world"]
    tok = Tokenizer.from_docs(docs)
    data = tok.to_dict()
    tok2 = Tokenizer.from_dict(data)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.chars == tok.chars


def test_microgpt_creation():
    model = MicroGPT(vocab_size=27, n_layer=1, n_embd=8, block_size=8, n_head=2)
    assert model.num_params > 0
    assert model.n_layer == 1
    assert model.n_embd == 8


def test_microgpt_forward():
    """Test a single forward pass produces logits."""
    model = MicroGPT(vocab_size=5, n_layer=1, n_embd=4, block_size=4, n_head=2)
    keys = [[] for _ in range(model.n_layer)]
    values = [[] for _ in range(model.n_layer)]

    logits = model.forward(0, 0, keys, values)
    assert len(logits) == 5  # vocab_size


def test_microgpt_train_and_generate():
    """Test end-to-end: train on tiny data, generate samples."""
    docs = ["abc", "bca", "cab", "abc", "bca", "cab"] * 10
    tok = Tokenizer.from_docs(docs)
    model = MicroGPT(
        vocab_size=tok.vocab_size,
        n_layer=1,
        n_embd=8,
        block_size=8,
        n_head=2,
        seed=42,
    )

    losses = model.train_on_docs(
        docs, tok, num_steps=50, learning_rate=0.01, log_every=0
    )

    assert len(losses) == 50
    # Loss should decrease
    assert losses[-1] < losses[0]

    samples = model.generate(tok, num_samples=5, temperature=0.8)
    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, str)


def test_microgpt_checkpoint(tmp_path):
    """Test save/load checkpoint."""
    docs = ["hello", "world", "test"] * 20
    tok = Tokenizer.from_docs(docs)
    model = MicroGPT(vocab_size=tok.vocab_size, n_layer=1, n_embd=8, block_size=8, n_head=2)
    model.train_on_docs(docs, tok, num_steps=10, log_every=0)

    # Save
    path = str(tmp_path / "test.json")
    model.save_checkpoint(path, tok, metadata={"test": True})

    # Load
    model2, tok2, meta = MicroGPT.load_checkpoint(path)
    assert model2.num_params == model.num_params
    assert tok2.vocab_size == tok.vocab_size
    assert meta["test"] is True

    # Generate from loaded model
    samples = model2.generate(tok2, num_samples=3)
    assert len(samples) == 3
