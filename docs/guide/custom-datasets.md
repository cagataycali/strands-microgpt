# Custom Datasets

Train on any text — names, poems, molecules, DNA, code, anything.

---

## From a List

```python
from strands_microgpt import MicroGPT, Tokenizer

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the bird sat on the word",
] * 100  # Repeat for more training data

tokenizer = Tokenizer.from_docs(docs)
model = MicroGPT(
    vocab_size=tokenizer.vocab_size,
    n_embd=32,
    block_size=32,
)

model.train_on_docs(docs, tokenizer, num_steps=2000)
samples = model.generate(tokenizer, num_samples=10, temperature=0.7)
```

## From a File

```python
model, tokenizer, docs = MicroGPT.from_dataset(
    dataset_path="my_data.txt"  # One document per line
)
model.train_on_docs(docs, tokenizer, num_steps=1000)
```

## From a URL

```python
model, tokenizer, docs = MicroGPT.from_dataset(
    dataset_url="https://example.com/poems.txt"
)
```

## Dataset Tips

!!! tip "Quality > Quantity"
    - One item per line (names, words, short phrases)
    - Keep items shorter than `block_size`
    - More repetition = faster learning
    - Character-level: works best with consistent patterns

## Examples of Custom Data

| Domain | Example Data | `block_size` |
|--------|-------------|--------------|
| Names | `alice`, `bob`, `charlie` | 16 |
| Words | `python`, `javascript`, `rust` | 16 |
| Molecules | `CCO`, `c1ccccc1`, `CC(=O)O` | 32 |
| DNA | `ATCGATCG`, `GCTAGCTA` | 32 |
| Phrases | `the quick brown fox` | 64 |

→ **Next:** [Tool Usage](tool-usage.md) | [Architecture](../architecture.md)
