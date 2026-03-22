# Examples

All examples are in the [`examples/`](https://github.com/cagataycali/strands-microgpt/tree/main/examples) directory.

| # | File | Description |
|---|------|-------------|
| 01 | [basic_training.py](https://github.com/cagataycali/strands-microgpt/blob/main/examples/01_basic_training.py) | Train on names, generate new ones |
| 02 | [strands_agent.py](https://github.com/cagataycali/strands-microgpt/blob/main/examples/02_strands_agent.py) | Use as a Strands Model provider |
| 03 | [tool_usage.py](https://github.com/cagataycali/strands-microgpt/blob/main/examples/03_tool_usage.py) | Train/generate via tool calls |
| 04 | [custom_dataset.py](https://github.com/cagataycali/strands-microgpt/blob/main/examples/04_custom_dataset.py) | Train on custom text data |
| 05 | [autograd_exploration.py](https://github.com/cagataycali/strands-microgpt/blob/main/examples/05_autograd_exploration.py) | Explore the autograd engine |

## Running

```bash
# Clone and install
git clone https://github.com/cagataycali/strands-microgpt.git
cd strands-microgpt
pip install -e .

# Run any example
python examples/01_basic_training.py
```
