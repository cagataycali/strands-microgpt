# Quickstart

Train a character-level GPT from scratch and generate text in under a minute.

---

## 1. Train on Names

```python
from strands_microgpt import MicroGPT

# Load dataset → build tokenizer → create model (one line)
model, tokenizer, docs = MicroGPT.from_dataset()

# Train (1000 steps on 32K names)
model.train_on_docs(docs, tokenizer, num_steps=1000)

# Generate
for name in model.generate(tokenizer, num_samples=10):
    print(name)
```

Expected output (after ~60s of training):
```
 1. kayla
 2. maren
 3. joline
 4. arian
 5. delia
```

---

## 2. Use as a Strands Agent

```python
from strands import Agent
from strands_microgpt import MicroGPTModel

model = MicroGPTModel(num_steps=500, temperature=0.5)
agent = Agent(model=model)
agent("Generate some names")
```

The model trains on first use, then generates text for every agent call.

---

## 3. Use as a Tool

```python
from strands import Agent
from strands_microgpt import microgpt_train, microgpt_generate

# Works with any model provider (Bedrock, OpenAI, etc.)
agent = Agent(tools=[microgpt_train, microgpt_generate])

agent("Train a MicroGPT on names for 500 steps")
agent("Generate 10 names with temperature 0.7")
```

---

## 4. Save & Load Checkpoints

```python
# Save after training
model.save_checkpoint("my_model.json", tokenizer)

# Load and generate later
model, tokenizer, meta = MicroGPT.load_checkpoint("my_model.json")
samples = model.generate(tokenizer, num_samples=20)
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_layer` | 1 | Transformer depth |
| `n_embd` | 16 | Embedding dimension |
| `block_size` | 16 | Context window |
| `n_head` | 4 | Attention heads |
| `num_steps` | 1000 | Training steps |
| `learning_rate` | 0.01 | Initial learning rate |
| `temperature` | 0.5 | Sampling temperature |

→ **Next:** [Autograd Engine](../guide/autograd.md) | [Custom Datasets](../guide/custom-datasets.md)
