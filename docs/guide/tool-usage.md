# Tool Usage

Use MicroGPT training and generation as tools inside any Strands agent.

---

## Available Tools

### `microgpt_train`

Train a model from scratch:

```python
from strands import Agent
from strands_microgpt import microgpt_train

agent = Agent(tools=[microgpt_train])
agent("Train a MicroGPT on the names dataset for 500 steps")
```

Parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `dataset_url` | names.txt | Training data URL |
| `dataset_path` | | Local file override |
| `num_steps` | 1000 | Training steps |
| `n_layer` | 1 | Transformer layers |
| `n_embd` | 16 | Embedding dim |
| `block_size` | 16 | Context window |
| `n_head` | 4 | Attention heads |
| `learning_rate` | 0.01 | Initial LR |
| `checkpoint_path` | | Save path |

### `microgpt_generate`

Generate from a trained checkpoint:

```python
from strands import Agent
from strands_microgpt import microgpt_generate

agent = Agent(tools=[microgpt_generate])
agent("Generate 10 names with temperature 0.7")
```

Parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `checkpoint_path` | /tmp/microgpt_checkpoint.json | Model path |
| `num_samples` | 20 | Samples to generate |
| `temperature` | 0.5 | Sampling temperature |

## Both Tools Together

```python
from strands import Agent
from strands_microgpt import microgpt_train, microgpt_generate

agent = Agent(tools=[microgpt_train, microgpt_generate])

# The agent can train then generate in one conversation
agent("Train a small GPT on names, then generate 10 creative names")
```

→ **Next:** [Architecture](../architecture.md) | [API Reference](../api-reference.md)
