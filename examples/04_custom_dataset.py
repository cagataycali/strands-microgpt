"""04: Train on custom data.

Train on any text file — poems, code, molecules, DNA sequences, etc.
"""

import tempfile
import time

from strands_microgpt.engine import MicroGPT, Tokenizer

print("=== 04: Custom Dataset ===")

# Create a tiny custom dataset
custom_data = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the bird sat on the word",
    "the fish sat in the dish",
    "the cat and the dog",
    "the bird and the fish",
    "the mat and the log",
    "the word and the dish",
] * 50  # Repeat for more training data

# Build tokenizer from our data
tokenizer = Tokenizer.from_docs(custom_data)
print(f"Vocab: {tokenizer.chars}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Create and train model
model = MicroGPT(
    vocab_size=tokenizer.vocab_size,
    n_layer=1,
    n_embd=32,
    block_size=32,
    n_head=4,
)

t0 = time.time()
losses = model.train_on_docs(
    custom_data, tokenizer, num_steps=2000, learning_rate=0.01, log_every=200
)

print(f"\nFinal loss: {losses[-1]:.4f}")
print(f"Train time: {time.time() - t0:.1f}s")

# Generate
print("\n--- Generated Phrases ---")
samples = model.generate(tokenizer, num_samples=10, temperature=0.7)
for i, s in enumerate(samples, 1):
    print(f"  {i:2d}. {s}")

# Save checkpoint
path = tempfile.mktemp(suffix=".json")
model.save_checkpoint(path, tokenizer)
print(f"\nCheckpoint saved: {path}")

print("=== PASS ===")
