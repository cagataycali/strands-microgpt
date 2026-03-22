# API Reference

## `Value`

Scalar-valued autograd node.

```python
from strands_microgpt import Value

a = Value(2.0)
```

### Operations

| Operation | Usage | Description |
|-----------|-------|-------------|
| `+` | `a + b` | Addition |
| `*` | `a * b` | Multiplication |
| `-` | `a - b` | Subtraction |
| `/` | `a / b` | Division |
| `**` | `a ** n` | Power |
| `.relu()` | `a.relu()` | ReLU activation |
| `.exp()` | `a.exp()` | Exponential |
| `.log()` | `a.log()` | Natural log |
| `.backward()` | `a.backward()` | Compute gradients |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.data` | `float` | The scalar value |
| `.grad` | `float` | The gradient (after `.backward()`) |

---

## `Tokenizer`

Character-level tokenizer with BOS token.

### Constructor

```python
Tokenizer(chars: List[str])
```

### Class Methods

| Method | Description |
|--------|-------------|
| `Tokenizer.from_docs(docs)` | Build from list of strings |
| `Tokenizer.from_dict(data)` | Deserialize from dict |

### Instance Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.encode(text)` | `List[int]` | Text → token IDs |
| `.decode(ids)` | `str` | Token IDs → text |
| `.to_dict()` | `Dict` | Serialize for saving |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.vocab_size` | `int` | Number of tokens (chars + BOS) |
| `.bos` | `int` | BOS token ID |
| `.chars` | `List[str]` | Character vocabulary |

---

## `MicroGPT`

Pure-Python GPT transformer.

### Constructor

```python
MicroGPT(
    vocab_size: int,
    n_layer: int = 1,
    n_embd: int = 16,
    block_size: int = 16,
    n_head: int = 4,
    seed: int = 42,
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.train_on_docs(docs, tokenizer, ...)` | `List[float]` | Train on documents, return losses |
| `.generate(tokenizer, ...)` | `List[str]` | Generate text samples |
| `.forward(token_id, pos_id, keys, values)` | `List[Value]` | Single-token forward pass |
| `.save_checkpoint(path, tokenizer, metadata)` | `None` | Save to JSON |

### Class Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.load_checkpoint(path)` | `(MicroGPT, Tokenizer, Dict)` | Load from JSON |
| `.from_dataset(url, path, **kwargs)` | `(MicroGPT, Tokenizer, docs)` | Load data + create model |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.num_params` | `int` | Total parameter count |
| `.n_layer` | `int` | Number of layers |
| `.n_embd` | `int` | Embedding dimension |

---

## `MicroGPTModel`

Strands Model provider interface.

```python
from strands_microgpt import MicroGPTModel

model = MicroGPTModel(
    dataset_url="...",
    num_steps=1000,
    n_layer=1,
    n_embd=16,
    temperature=0.5,
    num_samples=20,
)
```

Implements the full [Strands Model interface](https://strandsagents.com): `stream()`, `format_request()`, `format_chunk()`, etc.

---

## Tools

### `microgpt_train`

```python
microgpt_train(
    dataset_url: str = "...",
    dataset_path: str = "",
    num_steps: int = 1000,
    n_layer: int = 1,
    n_embd: int = 16,
    block_size: int = 16,
    n_head: int = 4,
    learning_rate: float = 0.01,
    seed: int = 42,
    checkpoint_path: str = "",
) -> Dict[str, Any]
```

### `microgpt_generate`

```python
microgpt_generate(
    checkpoint_path: str = "/tmp/microgpt_checkpoint.json",
    num_samples: int = 20,
    temperature: float = 0.5,
    seed: int = 0,
) -> Dict[str, Any]
```
