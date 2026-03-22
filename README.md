# strands-microgpt

[![PyPI version](https://badge.fury.io/py/strands-microgpt.svg)](https://pypi.org/project/strands-microgpt/)
[![CI](https://github.com/cagataycali/strands-microgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/cagataycali/strands-microgpt/actions/workflows/ci.yml)

**Karpathy's pure-Python microGPT as a [Strands Agents](https://strandsagents.com) model provider — zero dependencies, pure autograd, trained from scratch.**

Based on [@karpathy's atomic GPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): *"The most atomic way to train and run inference for a GPT in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency."*

---

## What is this?

A **complete GPT implementation** — autograd engine, transformer architecture, tokenizer, Adam optimizer, training loop, and inference — in **pure Python with zero dependencies**. No PyTorch, no NumPy, no CUDA. Just `math`, `random`, and the algorithm.

Packaged as a proper Strands model provider so you can:

1. **Use it as a Model** — drop-in Strands `Model` interface for character-level generation
2. **Use it as a Tool** — train and generate from any Strands agent via tool calls
3. **Learn from it** — the entire algorithm is readable, hackable, and documented

---

## Install

```bash
pip install strands-microgpt
```

> **Requirements:** Python ≥3.10, `strands-agents`. That's it. No GPU needed.

---

## Quick Start

### As a standalone engine (zero deps)

```python
from strands_microgpt import MicroGPT

# Load dataset, build tokenizer, create model
model, tokenizer, docs = MicroGPT.from_dataset()

# Train (1000 steps on names.txt)
model.train_on_docs(docs, tokenizer, num_steps=1000)

# Generate new names
for name in model.generate(tokenizer, num_samples=10):
    print(name)
```

### As a Strands Model provider

```python
from strands import Agent
from strands_microgpt import MicroGPTModel

model = MicroGPTModel(num_steps=1000, temperature=0.5)
agent = Agent(model=model)
agent("Generate some names")
```

### As a Tool (in any agent)

```python
from strands import Agent
from strands_microgpt import microgpt_train, microgpt_generate

# Use with Bedrock, OpenAI, or any model
agent = Agent(tools=[microgpt_train, microgpt_generate])
agent("Train a GPT on the names dataset for 500 steps, then generate 10 names")
```

---

## Architecture

The complete algorithm in ~300 lines:

```
strands_microgpt/
├── engine.py           # Value (autograd), Tokenizer, MicroGPT (transformer)
├── microgpt_model.py   # Strands Model interface
└── tools/
    ├── microgpt_train.py     # Training tool
    └── microgpt_generate.py  # Generation tool
```

### The Autograd Engine

```python
from strands_microgpt import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a  # builds computation graph
c.backward()   # backpropagate gradients
print(a.grad)  # 4.0 (dc/da = b + 1)
```

Supports: `+`, `*`, `-`, `/`, `**`, `relu()`, `exp()`, `log()`, `backward()`

### The Transformer

GPT-2 architecture with:
- **RMSNorm** (instead of LayerNorm)
- **No biases**
- **ReLU** (instead of GeLU)
- **Multi-head causal attention**
- **Adam optimizer** with linear LR decay

### Parameters

| Config | Default | Description |
|--------|---------|-------------|
| `n_layer` | 1 | Transformer depth |
| `n_embd` | 16 | Embedding dimension |
| `block_size` | 16 | Context window |
| `n_head` | 4 | Attention heads |
| `num_steps` | 1000 | Training steps |
| `learning_rate` | 0.01 | Initial LR (linear decay) |
| `temperature` | 0.5 | Generation temperature |

---

## Custom Datasets

Train on anything — names, poems, molecules, DNA, code:

```python
from strands_microgpt import MicroGPT, Tokenizer

docs = ["the cat sat on the mat", "the dog sat on the log"] * 100
tokenizer = Tokenizer.from_docs(docs)
model = MicroGPT(vocab_size=tokenizer.vocab_size, n_embd=32, block_size=32)

model.train_on_docs(docs, tokenizer, num_steps=2000)
samples = model.generate(tokenizer, num_samples=10, temperature=0.7)
```

### Checkpoints

```python
# Save
model.save_checkpoint("model.json", tokenizer)

# Load
model, tokenizer, metadata = MicroGPT.load_checkpoint("model.json")
samples = model.generate(tokenizer, num_samples=10)
```

---

## Examples

| Example | Description |
|---------|-------------|
| [01_basic_training.py](examples/01_basic_training.py) | Train on names, generate new ones |
| [02_strands_agent.py](examples/02_strands_agent.py) | Use as a Strands Model provider |
| [03_tool_usage.py](examples/03_tool_usage.py) | Train/generate via tool calls |
| [04_custom_dataset.py](examples/04_custom_dataset.py) | Train on custom text data |
| [05_autograd_exploration.py](examples/05_autograd_exploration.py) | Explore the autograd engine |

---

## Why?

> *"Everything else is just efficiency."* — @karpathy

This package exists to show that:

1. **A GPT is just math.** No magic, no black boxes. The entire algorithm fits in your head.
2. **Strands Model interface is universal.** If it can generate tokens, it can be a Strands model.
3. **Understanding > Using.** Train a transformer from scratch to truly grok what LLMs do.

The model is tiny and slow (pure Python, no vectorization). For production, use Bedrock, OpenAI, or any real provider. For **learning**, this is the best code to read.

---

## API Reference

### `MicroGPT`

```python
MicroGPT(vocab_size, n_layer=1, n_embd=16, block_size=16, n_head=4, seed=42)
```

- `.train_on_docs(docs, tokenizer, num_steps, learning_rate, log_every, callback)` → `List[float]`
- `.generate(tokenizer, num_samples, temperature, max_length)` → `List[str]`
- `.save_checkpoint(path, tokenizer, metadata)` → None
- `.load_checkpoint(path)` → `(MicroGPT, Tokenizer, Dict)` (classmethod)
- `.from_dataset(dataset_url, dataset_path, **kwargs)` → `(MicroGPT, Tokenizer, docs)` (classmethod)

### `MicroGPTModel` (Strands Model)

```python
MicroGPTModel(dataset_url, num_steps=1000, temperature=0.5, ...)
```

Drop-in replacement for any Strands model. Trains on first use, then generates.

### `Value` (Autograd)

```python
Value(data)  # scalar autograd node
```

Supports: `+`, `*`, `-`, `/`, `**`, `.relu()`, `.exp()`, `.log()`, `.backward()`

### Tools

- `microgpt_train(dataset_url, num_steps, n_layer, ...)` — Train a model
- `microgpt_generate(checkpoint_path, num_samples, temperature)` — Generate from checkpoint

---

## Resources

- [Karpathy's GPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — The original
- [micrograd](https://github.com/karpathy/micrograd) — Karpathy's autograd engine
- [makemore](https://github.com/karpathy/makemore) — Character-level language modeling
- [Strands Agents](https://strandsagents.com) — The agent framework
- [strands-cosmos](https://github.com/cagataycali/strands-cosmos) — NVIDIA Cosmos VLM provider

---

## License

MIT | Based on [@karpathy's work](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) | Built with [Strands Agents](https://strandsagents.com)
