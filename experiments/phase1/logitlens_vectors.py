"""
Project steering vectors through unembedding matrix to see what tokens they represent
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B"
VECTOR_PATH = "../thinking-llms-interp/train-vectors/results/vars/optimized_vectors/llama-3.1-8b_bias.pt"
TOP_K = 10

# Load
print(f"Loading model: {MODEL}")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print(f"Loading vector: {VECTOR_PATH}")
vector = torch.load(VECTOR_PATH, map_location = "cuda")
if isinstance(vector, dict):
    vector = next(iter(vector.values()))
vector = vector.to(model.dtype).to(model.device).squeeze()

# Project through unembedding
with torch.no_grad():
    logits = model.lm_head(vector)

# Top promoted tokens
top_vals, top_idx = logits.topk(TOP_K)
print(f"\n--- Promoted Tokens ---")
for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
    print(f"  {val:8.3f}  {repr(tokenizer.decode([idx]))}")

# Top suppressed tokens  
bot_vals, bot_idx = logits.topk(TOP_K, largest=False)
print(f"\n--- Suppressed Tokens ---")
for val, idx in zip(bot_vals.tolist(), bot_idx.tolist()):
    print(f"  {val:8.3f}  {repr(tokenizer.decode([idx]))}")
