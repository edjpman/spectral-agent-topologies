'''

[DEVELOPMENT ONLY]

Simple test to determine whether HF wiring to lightweight model is working.

'''

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Loading model....')


tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float32
)

print("Running test...")

inputs = tokenizer("Say hello", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0]))
