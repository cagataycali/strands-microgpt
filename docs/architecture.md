# Architecture

How strands-microgpt is structured internally.

---

## Package Structure

```
strands_microgpt/
├── __init__.py              # Exports: Value, MicroGPT, Tokenizer, MicroGPTModel, tools
├── engine.py                # Core: Value (autograd), Tokenizer, MicroGPT (transformer)
├── microgpt_model.py        # Strands Model interface
└── tools/
    ├── __init__.py           # Tool exports
    ├── microgpt_train.py     # Training tool (@tool decorated)
    └── microgpt_generate.py  # Generation tool (@tool decorated)
```

## Component Hierarchy

```mermaid
graph TD
    SM["strands.models.Model<br/><i>Abstract base class</i>"] --> MGM["MicroGPTModel<br/><i>Strands Model interface</i>"]
    MGM --> MG["MicroGPT<br/><i>Transformer</i>"]
    MG --> V["Value<br/><i>Autograd engine</i>"]

    style SM fill:#264653,color:#fff
    style MGM fill:#e65100,color:#fff
    style MG fill:#e65100,color:#fff
    style V fill:#e65100,color:#fff
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as Strands Agent
    participant Model as MicroGPTModel
    participant GPT as MicroGPT
    participant AG as Value (Autograd)

    User->>Agent: agent("Generate names")
    Agent->>Model: stream(messages)
    Model->>Model: _ensure_trained()
    Model->>GPT: generate(tokenizer, samples)
    GPT->>AG: forward(token, pos)
    AG->>AG: build computation graph
    AG->>GPT: logits
    GPT->>GPT: softmax → sample
    GPT->>Model: generated text
    Model->>Agent: StreamEvents
    Agent->>User: Result
```

## The Transformer

GPT-2 architecture (simplified):

- **Embeddings** — token + position
- **RMSNorm** — root mean square normalization (no LayerNorm)
- **Multi-Head Attention** — Q, K, V projections + causal mask
- **MLP** — 2-layer feedforward with ReLU
- **Residual connections** — add input to output
- **No biases** — weights only

```mermaid
graph TD
    E["Token + Position Embeddings"] --> N1["RMSNorm"]
    N1 --> A["Multi-Head Attention"]
    A --> R1["+ Residual"]
    E --> R1
    R1 --> N2["RMSNorm"]
    N2 --> M["MLP (ReLU)"]
    M --> R2["+ Residual"]
    R1 --> R2
    R2 --> LM["Language Model Head"]
    LM --> S["Softmax → Sample"]
```

## Two Usage Modes

```mermaid
graph TD
    subgraph "Mode 1: As the Model"
        A1["Agent(model=MicroGPTModel())"] --> B1["MicroGPT IS the brain"]
    end

    subgraph "Mode 2: As a Tool"
        A2["Agent(tools=[microgpt_train])"] --> B2["MicroGPT is a tool<br/>called by another model"]
    end

    style B1 fill:#e65100,color:#fff
    style B2 fill:#264653,color:#fff
```
