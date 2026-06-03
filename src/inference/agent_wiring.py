
'''
A wrapper for agent functionality. Includes:

- Local inference
- Dummy model 
- JSON extraction
- Context reset

Heavy models can be run with high-compute resources, whereas the local wiring test can be run on a laptop.


'''

import json
import re
import random
import time
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


## ----------
## Tests and Constants
## ----------

#Ensure the proper functionality exists
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


#Schema for strict rule following decoding
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "Reasoning": {"type":"string"},
        "Value": {"type": "number"},
        "Parity": {"type": "string", "enum": ["A", "B"]},
        "Level": {"type": "integer", "minimum": 1, "maximum": 9}
    },
    #Adding reasoning so the agent can think first if needed -- is remove before results are saved
    "required": ["Reasoning", "Value", "Parity", "Level"]
}


## ----------
## JSON HELPERS 
## ----------


def extract_json(text):
    '''
    Robust JSON parser. The function utilizes three different methods:

    - 1) Standard JSON loading

        If LLM follows exact instructions...
            {'Value': 74, 'Parity': 'Alpha'}
    
    - 2) Markdown Removing 

        If LLM adds its own formatting....
            ```json
                {'Value': 74, 'Parity': 'Alpha'}
            ```

    - 3) First {} extraction
        If LLM misses strict formatting...
            {'Value': 74, 'Parity': 'Alpha'}
            Hope that helps.

    - 4) Complete Failure
        If LLM completely misses...
            The answer is Value = 74
    
    '''
    #Try 1: Standard Load
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    #Try 2: MD Remover
    fenced = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    #Try 3: Extracts the largest JSON object
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    #Else None is returned
    return None



def majority_state(responses):
    '''
    Given a list of JSON dicts the most frequently occuring is kept. 
    Ties are broken by first encountered. 
    The mesh topology uses the function.
    
    '''
    if not responses:
        return {}
    #Treats each dict as its own string
    arranged = [json.dumps(r, sort_keys=True) for r in responses]
    counts = {}
    for a in arranged:
        #Creates count of dict string
        counts[a] =  counts.get(a,0) + 1
    #Selects the most common dict string and converts back to JSON
    majority = max(counts, key=counts.__getitem__)
    return json.loads(majority)



## ----------
## MOCK AGENT
## ----------

class MockAgnet:
    '''
    Simulates the real agent behavior to validate topology flow, and wiring code.
    
    '''

    def __init__(self, noise_prob=0.05):
        self.noise_prob = noise_prob #prob of returning malformed response
        self._history = [] #to simulate context


    def reset(self):
        '''
        To simulate resetting context during agent passes
        '''
        self._history.clear()


    def run(self, task, context):
        '''
        Lightly mutates JSON string to simulate a new version of the context.
        If context is a list (i.e. star mesh), the first entry is used.

        '''
        #Either the first item that is a dict...
        if isinstance(context, list):
            base = next(
                (c for c in context if isinstance(c, dict)), {}
                )
        #...the og dict if it exists...
        elif isinstance(context, dict):
            base = dict(context)
        #...or empty if other 
        else:
            base = {}

        state = dict(base)

        #Light manipulation of the state
        if 'Value' in state:
            try:
                state['Value'] = int(state['Value']) + 1
            except (ValueError, TypeError):
                state['Value'] = state['Value']
        if 'Parity' in state:
            state['Parity'] = 'Beta' if state['Parity'] == 'Alpha' else 'Alpha'
        if 'Level' in state:
            try:
                state['Level'] = int(state['Level']) + 1
            except (ValueError, TypeError):
                pass

        #To introduce some random failure component
        if random.random() < self.noise_prob:
            return 'Sorry I couldnt determine the state.'
        
        return json.dumps(state)





## ----------
## HF AGENT
## ----------

class HFAgent:
    '''
    Local HF model is loaded once, and context_reset() clears the prompt history.

    '''
    def __init__(self, model_name, device='cuda',max_new_tokens=30):
        if not _HF_AVAILABLE:
            raise ImportError('Transformers and torch are required for real agent!')
        
        print(f"HF Agent Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == 'cpu':
            model_kwargs = {}
        else:
            model_kwargs = {'device_map': device}

        #Boilerplate HF model load to CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            **model_kwargs
        )

        if device == 'cpu':
            self.model.to('cpu')

        #Boilerplate HF decoding
        self.pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        self._history = []
        print('HF Agent Model Ready')


    def reset(self):
        '''
        Forcefully clears context information if model were to have that ability. 

        '''
        self._history.clear()


    def run(self, task, context):
        '''
        Processes a task based on the provided context to generate a JSON state update.
        A stripped string containing the model's generated JSON response is returned.
        
        '''
        context_str = (
            json.dumps(context, indent=2)
            if context is not None and not isinstance(context, str)
            else context or ""
        )
        
        #Prompt scaffolding
        prompt = (
            f'''
            You are a precise JSON state machine. 
            Always respond with ONLY a valid JSON object {{'Value':int,'Parity':str,'Level':int}}
            Current State: {context_str}
            Task: {task}
            Please provide the updated JSON object:
            '''
        )
        print("[HFAgent] Sending prompt to model...")

        output = self.pipe(prompt)[0]['generated_text']

        print("[HFAgent] Model returned output")

        #Incase the model attempts to reprovide the entire prompt again
        if output.startswith(prompt):
            output = output[len(prompt):]
        return output.strip()





## ----------
## GENERAL AGENT
## ----------


class Agent:

    '''
    A meta execution class for the agents. Simplifies the complexity of calls needed to be made.

    '''

    def __init__(self, mock=False, model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0', 
                 device='cuda', noise_prob=0.05, enforce_eager=False,
                 max_tokens=128, use_guided_json=True, temperature=0.7, top_p=0.9
                 ):
        #General params to modify the experimentation process
        self.mock = mock
        self.model_name = model_name
        self.device = device
        self.enforce_eager = enforce_eager
        self.max_tokens = max_tokens
        self.use_guided_json = use_guided_json
        self.temperature = temperature
        self.top_p = top_p

        #Routes the model backend for the type of experiment
        if mock:
            #Uses mock agent for local testing
            self._backend = MockAgnet(noise_prob=noise_prob)
        elif device == 'cuda':
            #Uses efficient vLLM processes for large jobs
            print(f"--- Initializing vLLM with {model_name} ---")
            self.model = LLM(model=model_name, enforce_eager=self.enforce_eager, trust_remote_code=True)

            if self.use_guided_json:
                #Strict JSON decoding for model drifting if necessary
                print(f"--- Using Guided JSON Decoding (Max Tokens: {self.max_tokens}) ---")
                self.sampling_params = SamplingParams(
                                temperature=self.temperature, 
                                top_p=self.top_p, 
                                max_tokens=self.max_tokens, 
                                structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA)
                            )
            else:
                #Freeform decoding to allow the model to think
                print(f"--- Free-Form Generation (Max Tokens: ({self.max_tokens})) ---")
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )
            
            self._backend = None
        else:
            #Fallback to the original slow HF inference
            self._backend = HFAgent(model_name, device)


    def context_reset(self):
        '''
        Uses the reset() method to clear any of the context between agent steps in the topologies.  

        '''
        if self._backend is not None:
            self._backend.reset()


    def execute_task(self, task, context):
        '''
        Runs a single inference step. 

        Inputs:
        - task: natural-language instructions for the step
        - context: the current JSON state dict or proposal dicts if needed for the topology

        Outputs:
        - Parsed JSON dict of the state. 
        - Input (previous) context if a parse failure occurs to keep loop from crashing on bad responses.
        
        '''
        #General task execution tied to agent backend with answer parsed
        print("[Agent] Executing task...")
        print(f"[Agent] Context: {context}")

        #Normalize context
        context_str = (
            json.dumps(context, indent=2)
            if context is not None and not isinstance(context, str)
            else context or ""
        )

        start_be = time.time()

        #Optimized vLLM inference here
        if self.device == 'cuda' and not self.mock:
            prompt = (f"""
            <|im_start|>system
            You are a state-tracking engine.
            Output ONLY valid JSON.
            Do not include markdown fences or extra text.
            <|im_end|>
            <|im_start|>user
            Apply the rules to the current state.

            Current State: {context_str}
            Task: {task}

            Return exactly:
            {{"Reasoning": "step-by-step reasoning here", "Value": float, "Parity": "A/B", "Level": int}}
            <|im_end|>
            <|im_start|>assistant
            {{"Reasoning": 
            """)
            #Single vLLM call + first completion 
            outputs = self.model.generate([prompt], self.sampling_params)
            raw = outputs[0].outputs[0].text
            if prompt in raw:
                raw = raw.split(prompt, 1)[-1]

            raw = raw.strip()
        else:
            #The old HF or Mock inference
            raw = self._backend.run(task=task, context=context_str)

        
        print(f"[Timing] Step took {time.time() - start_be:.2f}s")

        #Truncated to not bloat CLI
        print(f"[Agent] Raw output: {raw[:200]}")

        parsed = extract_json(raw)

        #Notification if bad data is produced
        if parsed is None:
            fallback = context if isinstance(context, dict) else {}
            print('Warning - could not parse the result. Returning previous value state forward.\n'
                  f'Raw output:{raw}'
                  )
            return fallback
        
        #Removes reasoning for clean downstream eval
        parsed.pop("Reasoning",None)
        
        return parsed


    def execute_batch(self, tasks, contexts):
        '''
        Runs a batched inference for parallel processing.
        
        '''
        print(f"[Agent] Executing batch of {len(tasks)} tasks...")
        start_be = time.time()

        prompts = []
        #Creates a prompt per task-pair
        for task, context in zip(tasks, contexts):
            context_str = (
                json.dumps(context, indent=2)
                if context is not None and not isinstance(context, str)
                else context or ""
            )

            prompt = (f"""
            <|im_start|>system
            You are a state-tracking engine.
            Output ONLY valid JSON.
            Do not include markdown fences or extra text.
            <|im_end|>
            <|im_start|>user
            Apply the rules to the current state.

            Current State: {context_str}
            Task: {task}

            Return exactly:
            {{"Reasoning": "step-by-step reasoning here", "Value": float, "Parity": "A/B", "Level": int}}
            <|im_end|>
            <|im_start|>assistant
            {{"Reasoning": 
            """)
            prompts.append(prompt)

        parsed_results = []

        if self.device == 'cuda' and not self.mock:
            #Batch inference through vLLM -- parallel decoding streams
            outputs = self.model.generate(prompts, self.sampling_params)

            #Parse each branch result independently while preserving order
            for i, output in enumerate(outputs):
                raw = output.outputs[0].text
                if prompts[i] in raw:
                    raw = raw.split(prompts[i], 1)[-1]

                parsed = extract_json(raw.strip())
                if parsed:
                    #Removes reasoning for downstream eval
                    parsed.pop("Reasoning", None)
                    parsed_results.append(parsed)
                else:
                    parsed_results.append(contexts[i] if isinstance(contexts[i], dict) else {})
        else:
            #Falls back in single-task call if needed
            for task, context in zip(tasks, contexts):
                parsed_results.append(self.execute_task(task, context))

        print(f"[Timing] Batch step took {time.time() - start_be:.2f}s")
        return parsed_results





