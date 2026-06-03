

'''
A high level orchestration layer. Experiments include: 
- Single experiment
- Boostrap
- All topologies 

Parallelization methodologies will be utilized to ensure performance when calling the LLM.

'''


import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm
 
from agents.topology import TopologyResult, chain, mesh, star
from inference.agent_wiring import Agent
from eval.eval_metrics import full_eval_trial, EvalResult




## ----------
## Defaults
## ----------

#Enhance logic to ensure model fails "productively"
ENHANCED_DIFFICULTY_PROMPTS = [
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Rules are dependent on each other.
        Do not skip ahead. Each rule requires the result of the previous.

        RULE 1 — Compute V_raw (do NOT round yet):
        If Parity is "A": V_raw = (Value * 1.25) + (Level * 2)
        If Parity is "B": V_raw = (Value * 0.75) - (Level * 2)
        Do not round V_raw. Carry the full decimal forward into Rule 2.

        RULE 2 — Compute V_new and P_new using V_raw from Rule 1:
        Take the decimal portion of V_raw only (i.e. V_raw minus its integer part).
        If that decimal portion is >= 0.5:
            V_new = round(V_raw + 1.5, 2)
        If that decimal portion is < 0.5 but > 0.0:
            V_new = round(V_raw - 0.5, 2)
        If the decimal portion is exactly 0.0:
            V_new = round(V_raw * 1.1, 2)
        Then compute P_new:
        If V_new < 70:  P_new = "A"
        If V_new >= 70: P_new = "B"

        RULE 3 — Compute L_new using BOTH V_new from Rule 2 AND P_new from Rule 2:
        Apply EXCEPT logic carefully:
        Apply L_new = Level + 2 (max 9) in ALL of the following cases EXCEPT
        when P_new is "B" and V_new is between 60 and 90 inclusive,
        in which case apply L_new = Level - 1 (min 1) instead.
        The remaining cases are:
        If Level >= 4 and V_new < 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new < 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    )
]

#Standard for state tracking task
DEFAULT_PROMPTS = [
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    ),
    (
        '''
        [STRICT INSTRUCTION]
        You are a JSON-only processor. Use the 'Reasoning' field to 
        calculate the new state before providing the final values.
        
        STATE SCHEMA:
        {
          "Reasoning": "Step-by-step math",
          "Value": <float>, 
          "Parity": "<A or B>", 
          "Level": <integer>
        }

        RULESET — apply in this exact order. Each pass of these rules equates to a single complete step:

        RULE 1 — Compute V_new:
        If Parity is "A": V_new = (Value * 1.25)
        If Parity is "B": V_new = (Value * 0.8)
        Round V_new to 2 decimal places.

        RULE 2 — Compute P_new:
        Take the integer part of V_new (ignore decimals).
        If that integer divided by 3 has remainder 0: P_new = "A"
        Otherwise: P_new = "B"

        RULE 3 — Compute L_new:
        If Level >= 4 and V_new > 60:  L_new = Level + 2  (maximum value is 9)
        If Level >= 4 and V_new <= 60: L_new = Level - 1  (minimum value is 1)
        If Level < 4  and V_new > 60:  L_new = Level + 1  (maximum value is 9)
        If Level < 4  and V_new <= 60: L_new = Level - 2  (minimum value is 1)

        FINAL OUTPUT FORMAT:
        - {"Reasoning": "step-by-step math", "Value": <float>, "Parity": "<A or B>", "Level": <integer>}
        '''
    )
]


DEFAULT_INITIAL_STATE = {'Value':74,'Parity':'A','Level':3}


ENHANCED_DIFFICULTY_GT = [
    {'Value':100.00,'Parity':'B','Level':5},
    {'Value':71.50,'Parity':'B','Level':4},
    {'Value':47.12,'Parity':'A','Level':6},
    {'Value':72.40,'Parity':'B','Level':5},
    {'Value':43.80,'Parity':'A','Level':7},
    {'Value':70.25,'Parity':'B','Level':6},
    {'Value':42.19,'Parity':'A','Level':8},
    {'Value':70.24,'Parity':'B','Level':7},
    {'Value':40.18,'Parity':'A','Level':9},
    {'Value':67.72,'Parity':'A','Level':9},
    {'Value':104.15,'Parity':'B','Level':9},
    {'Value':59.61,'Parity':'A','Level':9}
]


DEFAULT_GROUND_TRUTH = [
    {'Value':92.50,'Parity':'B','Level':3},
    {'Value':74.00,'Parity':'B','Level':4},
    {'Value':59.20,'Parity':'B','Level':6},
    {'Value':47.36,'Parity':'B','Level':5},
    {'Value':37.89,'Parity':'B','Level':4},
    {'Value':30.31,'Parity':'A','Level':3},
    {'Value':37.89,'Parity':'B','Level':1},
    {'Value':30.31,'Parity':'A','Level':1},
    {'Value':37.89,'Parity':'B','Level':1},
    {'Value':30.31,'Parity':'A','Level':1},
    {'Value':37.89,'Parity':'B','Level':1},
    {'Value':30.31,'Parity':'A','Level':1}
]



## ----------
## Perturbation helpers
## ----------

def perturb_state(state, mode='numeric', target_key='Value', epsilon=0.5, noise_key='Weather', noise_val='Sunny'):
    '''
    Returns a copy pf the perturbed version of the state. 
    
    ''' 
    perturbed = copy.deepcopy(state)
    
    #If a numeric perturbation the new numeric value is added to the existing
    if mode == 'numeric':
        try:
            perturbed[target_key] = float(perturbed[target_key]) + epsilon
        except (KeyError, TypeError, ValueError):
            pass

    #If a noise field then the noise value is added
    elif mode == 'noise_field':
        perturbed[noise_key] = noise_val
    return perturbed



## ----------
## Topology Runner
## ----------

def _run_topology(topology, agent, prompts, init_state, n_leaf, n_mesh, rules_text):
    '''
    Routes the topologies for the experiment run.

    '''
    if topology == 'chain':
        return chain(agent, prompts, init_state)
    if topology == 'star':
        return star(agent, prompts, init_state, n_leaf_agents=n_leaf, rules_text=rules_text)
    if topology == 'mesh':
        return mesh(agent, prompts, init_state, n_agents=n_mesh, rules_text=rules_text)
    raise ValueError(f'Unknown topology: {topology}!')


def _inject_ground_truth(result, ground_truth):
    '''
    Backfills the ground truth into each StepRecord for eval purposes.

    As a conceptual example...

    - Say at step 2 the topo output is {'Value': 16}
    - And at step 2 the ground truth is {'Value': 17}
    - The new StepRecord data object would have both at step 2
    
    '''
    for rec, gt in zip(result.step_records, ground_truth):
        rec.ground_truth = gt



## ----------
## Eval
## ----------


def _trial_worker(args):
    '''
    Executes one trial of the experiment pipeline.

    - This includes:
        - Baseline topology result
        - Perturbed result
    
    '''

    #Extracts the configuration out of the argument dictionary.
    agent = args['agent'] #receives agent for re-loading issue mentioned in main_rnnr.py
    topology = args['topology']
    trial_index = args['trial_index']
    prompts = args['prompts']
    initial_state = args['initial_state']
    ground_truth = args['ground_truth']
    mock = args['mock']
    model_name = args['model_name']
    device = args['device']
    #n_agents = args['n_agents'] #Only for when agent counts across all topologies are the same
    n_leaf = args['n_leaf']
    n_mesh = args['n_mesh']
    noise_prob = args['noise_prob']
    perturb_mode = args['perturb_mode']
    epsilon = args.get('epsilon', 0.5)
    value_key = args['value_key']
    enforce_eager = args.get('enforce_eager', False)
    rules_text = args.get('rules_text', "")

    #Instantiates the agent
    #Removed for re-loading issue mentioned in main_rnnr.py
    #agent = Agent(mock=mock, model_name=model_name, device=device, noise_prob=noise_prob, enforce_eager=enforce_eager)

    print(f"[Trial {trial_index}] Starting baseline run using the shared model")

    #Baseline run
    #Using separate params for the counts of leaf and mesh agents
    print(f"[Trial {trial_index}] Starting baseline run")
    baseline = _run_topology(topology, agent, prompts, initial_state, n_leaf=n_leaf, n_mesh=n_mesh, rules_text=rules_text)
    print(f"[Trial {trial_index}] Baseline complete")
    _inject_ground_truth(baseline, ground_truth)

    #Perturbed run
    #Using separate params for the counts of leaf and mesh agents 
    perturbed_init = perturb_state(initial_state, mode=perturb_mode, epsilon=epsilon)
    perturbed = _run_topology(topology, agent, prompts, perturbed_init, n_leaf=n_leaf, n_mesh=n_mesh, rules_text=rules_text)

    #Empirical evaluation 
    eval_results = full_eval_trial(
        step_records=baseline.step_records,
        ground_truth=ground_truth,
        baseline_final=baseline.final_state,
        perturbed_final=perturbed.final_state,
        value_key=value_key
    )

    #Convert to dict form
    def _eval_to_dict(e):
        if e is None:
            return None
        return {
            'metric':e.metric, 
            'topology':e.topology, 
            'score':e.score, 
            'per_step':[
                asdict(x) if hasattr(x,'__dataclass_fields__') else x
                for x in e.per_step
            ], 
            'notes':e.notes
        }
    
    print(type(baseline.final_state))
    print(type(perturbed.final_state))
    
    #Returns output of the experiment in a JSON summary
    return {
        'trial_index': trial_index,
        'topology': topology,
        'baseline_final': baseline.final_state,
        'perturbed_final': perturbed.final_state,
        'step_records': [asdict(sr) for sr in baseline.step_records],
        'eval': {
            'spectral_radius': _eval_to_dict(eval_results['spectral_radius']),
            'spectral_gap': _eval_to_dict(eval_results['spectral_gap']),
            'condition_number': _eval_to_dict(eval_results['condition_number'])
        }
    }




## ----------
## Experiment Execution
## ----------


def run_experiment(topology, prompts=None, initial_state=None, ground_truth=None, 
                   mock=False, model_name='meta-llama', device='cuda', n_leaf=5, n_mesh=4, 
                   noise_prob= 0.05, perturb_mode='numeric', value_key='Value', enforce_eager=False
    ):
    '''
    Runs a single trial for one topology and returns a JSON summary of the results.

    '''
    return _trial_worker({
        'topology': topology,
        'trial_index': 0,
        'prompts': prompts or DEFAULT_PROMPTS,
        'initial_state': initial_state or DEFAULT_INITIAL_STATE,
        'ground_truth': ground_truth or DEFAULT_GROUND_TRUTH,
        'mock': mock,
        'model_name': model_name,
        'device': device,
        #'n_agents': n_agents, #Only for when agent counts across all topologies are the same
        'n_leaf':n_leaf,
        'n_mesh':n_mesh,
        'noise_prob': noise_prob,
        'perturb_mode': perturb_mode,
        'value_key': value_key,
        'enforce_eager': enforce_eager
    })


def run_bootstrap(topology, agent=None, n_trials=30, prompts=None, initial_state=None, 
                    ground_truth=None, mock=False, model_name='meta-llama', 
                    device='cuda', n_leaf=5, n_mesh=4, 
                    noise_prob=0.05, perturb_mode='numeric', epsilon=0.5,
                    value_key='Value', n_workers=1, enforce_eager=False, task_difficulty='default'
    ):
    '''
    Runs a boostrap trial for one topology. Trials are parallelised for efficiency of experimentation.

    Use n_workers=1 for debugging or if non-parallelised is suffice.
    
    '''

    #Sets defaults for experimentation 
    initial_state = initial_state or DEFAULT_INITIAL_STATE

    if task_difficulty == 'enhanced':
        prompts = prompts or ENHANCED_DIFFICULTY_PROMPTS
        ground_truth = ground_truth or ENHANCED_DIFFICULTY_GT
    else:
        prompts = prompts or DEFAULT_PROMPTS
        ground_truth = ground_truth or DEFAULT_GROUND_TRUTH

    current_rules = prompts[0]

    #Creates a list of dicts of the indexed trial runs for execution
    trial_args = [
        {
            'rules_text': current_rules,
            'agent': agent,
            'topology': topology,
            'epsilon': epsilon,
            'trial_index': i,
            'prompts': prompts,
            'initial_state': initial_state,
            'ground_truth': ground_truth,
            'mock': mock,
            'model_name': model_name,
            'device': device,
            #'n_agents': n_agents, #Only for when agent counts across all topologies are the same
            'n_leaf':n_leaf,
            'n_mesh':n_mesh,
            'noise_prob': noise_prob,
            'perturb_mode': perturb_mode,
            'value_key': value_key,
            'enforce_eager': enforce_eager
        }
        for i in range(n_trials)
    ]

    results = []

    #Executes a non-parallelised version of the experiment, stores the results, and prints summary
    if n_workers == 1:
        for args in tqdm(trial_args, desc=f"Bootstrap: {topology}", unit="trial"): #formerly just trial_args
            res = _trial_worker(args)
            results.append(res)
            sr = res['eval']['spectral_radius']
            sg = res['eval']['spectral_gap']
            cn = res['eval']['condition_number']
            print(
               f"{topology} trial {res['trial_index']+1}/{n_trials}"
               f"SR={sr['score'] if sr else 'N/A'}"
               f"SG={sg['score'] if sg else 'N/A'}"
               f"CN={cn['score'] if cn else 'N/A'}" 
            )
    #Executes parallelised version of the experiment, stores the results, and prints trial info
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_trial_worker, a): a['trial_index'] for a in trial_args}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Bootstrap: {topology}", unit="trial"): #formerly as_completed(futures)
                #Returns what _trial_worker() would
                res = future.result()
                results.append(res)
                print(
                f"{topology} trial {res['trial_index']+1}/{n_trials} done"
                )
    results.sort(key=lambda r: r['trial_index'])
    return results



def run_all_topologies(
        agent=None, n_trials=30, prompts=None, initial_state=None, 
        ground_truth=None, mock=False, model_name='meta-llama', 
        device='cuda', n_leaf=5, n_mesh=4, 
        noise_prob=0.05, perturb_mode='numeric', epsilon=0.5,
        value_key='Value', n_workers=1, enforce_eager=False, task_difficulty='default'
    ):
    '''
    Runs bootstrap over all three topologies.
    
    '''
    all_results = {}

    #Loops through the various topologies executing the boostrap method at each pass
    for topo in ('chain','star','mesh'):
        print(f"\n{'='*52}")
        print(f'Topology: {topo.upper()} ({n_trials} trials, {n_workers} workers)')
        print(f"{'='*52}")

        all_results[topo] = run_bootstrap(
            topology=topo,
            agent=agent,
            n_trials=n_trials,
            prompts=prompts,
            initial_state=initial_state,
            ground_truth=ground_truth,
            mock=mock,
            model_name=model_name,
            device=device,
            n_leaf=n_leaf,
            n_mesh=n_mesh,
            noise_prob=noise_prob,
            perturb_mode=perturb_mode,
            epsilon=epsilon,
            value_key=value_key,
            n_workers=n_workers,
            enforce_eager=enforce_eager,
            task_difficulty=task_difficulty
        )
    return all_results


def save_results(results, filename, data_dir):
    '''
    Results are saved as a JSON object.
    
    '''
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/filename

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Written --> {out_path}')

    return out_path






