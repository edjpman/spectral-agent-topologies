
from vllm import LLM, SamplingParams
import os

MODEL_PATH = "Qwen/Qwen3-8B"

def test_load():
    print("Checking vLLM compatibility with Qwen3-8B...")
    #Initializes a heavy model for testing
    try:
        llm = LLM(model=MODEL_PATH, max_model_len=1024, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        
        output = llm.generate("Hello, are you functional?", sampling_params)
        print(f"Success! Response: {output[0].outputs[0].text}")
    except Exception as e:
        print(f"GPU Test Failed: {e}")

if __name__ == "__main__":
    test_load()

