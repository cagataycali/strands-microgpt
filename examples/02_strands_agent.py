"""02: Use MicroGPT as a Strands Agent model provider.

The agent uses a tiny from-scratch GPT for generation.
Educational — shows the Strands Model interface with the simplest possible model.
"""

import time

from strands import Agent

from strands_microgpt import MicroGPTModel

print("=== 02: MicroGPT as Strands Model ===")
t0 = time.time()

model = MicroGPTModel(
    num_steps=500,   # Fewer steps for demo speed
    n_layer=1,
    n_embd=16,
    temperature=0.5,
    num_samples=10,
)

agent = Agent(model=model)
result = agent("Generate some names")

print(f"\nTime: {time.time() - t0:.1f}s")
print("=== PASS ===")
