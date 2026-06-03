'''

[DEVELOPMENT ONLY]

Simple test to determine the efficiency of decoding inference through vLLM specifically.

'''

from vllm import LLM, SamplingParams

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
sampling_params = SamplingParams(temperature=0, max_tokens=30)

outputs = llm.generate(["The spectral radius of a graph is"], sampling_params)

for output in outputs:
    print(f"Generated text: {output.outputs[0].text}")