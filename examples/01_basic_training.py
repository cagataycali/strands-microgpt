"""01: Basic training and generation.

Train a character-level GPT on names, then generate new ones.
Zero dependencies beyond Python stdlib.
"""

import time

from strands_microgpt.engine import MicroGPT

print("=== 01: Train & Generate Names ===")
t0 = time.time()

# Load dataset, build tokenizer, create model — one line
model, tokenizer, docs = MicroGPT.from_dataset()

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Params: {model.num_params}")
print(f"Docs: {len(docs)}")

# Train
losses = model.train_on_docs(
    docs, tokenizer, num_steps=1000, learning_rate=0.01, log_every=100
)
print(f"\nFinal loss: {losses[-1]:.4f}")

# Generate
print("\n--- Generated Names ---")
samples = model.generate(tokenizer, num_samples=20, temperature=0.5)
for i, name in enumerate(samples, 1):
    print(f"  {i:2d}. {name}")

print(f"\nTotal time: {time.time() - t0:.1f}s")
print("=== PASS ===")
