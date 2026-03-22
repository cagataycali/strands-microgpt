"""03: Use MicroGPT as a tool inside another agent.

Train and generate using tool calls from any Strands agent.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands import Agent

from strands_microgpt import microgpt_generate, microgpt_train

print("=== 03: MicroGPT as Tool ===")

# Use with any model provider (Bedrock, OpenAI, etc.)
agent = Agent(tools=[microgpt_train, microgpt_generate])

# Train a model
agent("Train a MicroGPT on the default names dataset for 500 steps")

# Generate from checkpoint
agent("Generate 10 names from the trained model with temperature 0.7")

print("=== PASS ===")
