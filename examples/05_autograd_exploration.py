"""05: Explore the autograd engine.

Understand backpropagation by building computation graphs manually.
"""

from strands_microgpt.engine import Value

print("=== 05: Autograd Engine ===")

# Simple computation graph
a = Value(2.0)
b = Value(3.0)
c = a * b         # c = 6
d = c + a          # d = 8
e = d * b          # e = 24

print(f"a = {a.data}, b = {b.data}")
print(f"c = a * b = {c.data}")
print(f"d = c + a = {d.data}")
print(f"e = d * b = {e.data}")

# Backpropagate
e.backward()

print("\nGradients (de/d...):")
print(f"  de/da = {a.grad}")  # de/da = b*b + b = 3*3 + 3 = 12
print(f"  de/db = {b.grad}")  # de/db = a*b + c+a = 2*3 + 6+2 = 14

# Neural network building blocks
print("\n--- ReLU ---")
for x in [-2, -1, 0, 1, 2]:
    v = Value(float(x))
    r = v.relu()
    print(f"  relu({x}) = {r.data}")

# Softmax (manual)
print("\n--- Softmax ---")
logits = [Value(1.0), Value(2.0), Value(3.0)]
max_val = max(v.data for v in logits)
exps = [(v - max_val).exp() for v in logits]
total = sum(exps)
probs = [e / total for e in exps]
print(f"  softmax([1,2,3]) = [{', '.join(f'{p.data:.4f}' for p in probs)}]")
print(f"  sum = {sum(p.data for p in probs):.4f}")

print("\n=== PASS ===")
