

'''
Evaluation metrics for the LLM topology experiments. 
These are directly relatable back to the initial spectral analysis, and include:

- Cumulative error - Spectral Radius 
- Average log convergence rate - Spectral Gap 
- Final state sensitivity - Condition Number

The methods leverage the StepRecord list from the topologies and ground truth to compute the results. 


'''

import math
import itertools
from dataclasses import dataclass
from agents.topology import StepRecord
from typing import Optional


## ----------
## Helpers 
## ----------

def _numeric(state, key):
    '''
    Extracts the numeric value from the state dict for a given key.
    Returns None if a missing key or the value is non-numeric. 

    '''
    val = state.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pairwise_disagreement(states, key):
    '''
    Extracts the quantitative feature and returns None is less than 2 values available.
    
    Essentially measures on average how different are the values from each other.

    - A list of quant meaures is obtained; i.e. [0.5 , 0.3, 0.2]

    - The combination of agent-pairs are created; i.e. (0.2, 0.3) then obtaining the disagreement 0.1

    - The total disagreement is then summed and multiplied by 2 since combinations() removes duplicates.

    - The value is then normalized by; total/N(N-1)
    
    '''
    #Extracts the numeric features from the dict states
    values = [_numeric(s,key) for s in states if _numeric(s,key) is not None]

    n = len(values)
    if n<2:
        return None
    
    #Agent vs agent alignment check
    total = sum(abs(a-b) for a, b in itertools.combinations(values, 2))

    #Returns the counts of ordered pairs 
    return (2*total)/(n*(n-1))


## ----------
## Return Data Class 
## ----------


@dataclass
class EvalResult:
    metric: str
    topology: str
    score: Optional[float]
    per_step: list
    notes: str = ""


## ----------
## Metric 1: Spectral Radius (cumulative error)
## ----------

def spectral_radius_eval_score(step_records, ground_truth, value_key):
    '''
    Takes the cumulative deviation of the quantitative measures at each step.
    A higher score means more deviation from the ground truth.

    Inputs:
    - step_records: outputs stored from TopologyResult.step_records
    - ground_truth: list of correct state dicts for each step
    - value_key: which key to compare numerically
    
    '''
    if len(step_records) != len(ground_truth):
        raise ValueError(
            f'Step record mismatch! Step Len {len(step_records)}, GT Len {len(ground_truth)}.'
        )
    
    #Gets numeric value of the predicted vs ground truth
    per_step = []
    for rec, gt in zip(step_records, ground_truth):
        pred_val = _numeric(rec.final_state, value_key)
        gt_val = _numeric(gt, value_key)

        if pred_val is None or gt_val is None:
            per_step.append(None)
        else:
            per_step.append(abs(pred_val-gt_val))

    #Computes the cumulative sum of the disagreement over steps
    valid = [v for v in per_step if v is not None]
    cumulative = sum(valid) if valid else None

    #Notes for QA
    notes = ''
    if len(valid) < len(per_step):
        missing = len(per_step) - len(valid)
        notes = f'{missing} step(s) skipped due to missing or non-numeric results. Key: {value_key}'

    #Stores result in eval data object
    return EvalResult(
        metric='spectral_radius',
        topology=step_records[0].topology if step_records else 'unknown',
        score=cumulative,
        per_step=per_step,
        notes=notes
    )



## ----------
## Metric 2: Spectral Gap (convergence rate)
## ----------


def spectral_gap_eval_score(step_records, ground_truth, value_key='Value'):
    '''
    Computes the average log convergence rate based on interagent disagreement at each step.

    Interagent Disagreement by Topology Computed as: 

    - Chain: Agent to agent difference 
    - Star: Average of how different each leaf agent is from each other
    - Mesh: Average of how different each mesh agent is from each other at pass 1
    
    '''
    topology = step_records[0].topology if step_records else 'unknown'

    #To compute D_t at each step
    D_series = []

    #Gets numeric value of the predicted vs ground truth
    #returns single disagreement value
    for rec, gt in zip(step_records, ground_truth):
        if topology == 'chain':
            pred_val = _numeric(rec.final_state, value_key)
            gt_val = _numeric(gt, value_key)
            if pred_val is None or gt_val is None:
                D_series.append(None)
            else:
                D_series.append(abs(pred_val-gt_val))

        else:
            #Star leverages leaf proposals and mesh leverages pass-1 proposals
            #returns single disagreement value
            agent_states = rec.proposals
            d = _pairwise_disagreement(agent_states, value_key)
            D_series.append(d)

    #For the rate as r_t = log(D_t+1 / D_t) for consecutive valid pairs
    per_step_rates = []
    notes_parts = []

    #Loops through each value in the D_t list
    for t in range(len(D_series) - 1):
        d_curr = D_series[t]
        d_next = D_series[t+1]

        #If one consecutive value is None a rate of None is returned
        if d_curr is None or d_next is None:
            per_step_rates.append(None)
            notes_parts.append(f'Step {t}-->{t+1}: skipped (None D value).')
            continue

        #If perfect agreement then 0 since convergence is already reached
        if d_curr == 0 and d_next == 0:
            per_step_rates.append(0.0)
            continue

        #If agents jump from perfect convergence then return None
        #Can't divide by 0
        if d_curr == 0:
            per_step_rates.append(None)
            notes_parts.append(f'Step {t}-->{t+1}: D_t=0, log undefined.')
            continue

        #Computes the log of the rate
        rate = math.log(d_next/d_curr) if d_next > 0 else math.log(1e-9/d_curr)
        per_step_rates.append(rate)

    #Computes the avg of the log convergence rates 
    valid_rates = [r for r in per_step_rates if r is not None]
    avg_rate = (sum(valid_rates)/len(valid_rates)) if valid_rates else None

    #Stores result in eval data object
    return EvalResult(
        metric='spectral_gap',
        topology=topology,
        score=avg_rate,
        per_step=per_step_rates,
        notes=' | '.join(notes_parts) if notes_parts else ''
    )



## ----------
## Metric 3: Condition Number (perturbation fragility)
## ----------


def condition_number_eval_score(baseline_final, perturbed_final, value_key='Value'):
    '''
    Compares two completed topology runs:
    - 1. Clean baseline
    - 2. Perturned initial state 

    Topology agnostic since the perterbation is applied in the experiment execution phase in the initial state dict.
    A higher score indicates a more fragile (brittle) topology. 

    The final state of the two runs are used ONLY for the score computation.

    '''
    #Extracts the quantitative figure from both final states
    x_t = _numeric(baseline_final, value_key)
    xp_t= _numeric(perturbed_final, value_key)


    #If the agent fails at the final state a None score is return, else the normal calculation    
    notes = ''
    if x_t is None or xp_t is None:
        notes = f"Could not extract numeric '{value_key}' from one or both final states."
        score = None
    else:
        score = abs(xp_t-x_t)

    #Stores result in eval data object
    return EvalResult(
        metric='condition_number',
        topology='',
        score=score,
        per_step=[score],
        notes=notes
    )




## ----------
## Collective Run 
## ----------


def full_eval_trial(step_records, ground_truth, perturbed_final, baseline_final, value_key):
    '''
    Runs all applicable metrics for a single trial and returns a dict of the results. 
    
    '''

    results = {}

    results['spectral_radius'] = spectral_radius_eval_score(
        step_records, ground_truth, value_key
    )

    results['spectral_gap'] = spectral_gap_eval_score(
        step_records, ground_truth, value_key
    )

    if perturbed_final is not None and baseline_final is not None:
        cn = condition_number_eval_score(
            baseline_final, perturbed_final, value_key
        )
        cn.topology = step_records[0].topology if step_records else 'unknown'
        results['condition_number'] = cn
    else:
        results['condition_number'] = None

    return results







