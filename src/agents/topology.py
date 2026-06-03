

'''
Implements the three multi-agent communication topologies:

- Chain: the telephone
- Star: the judge
- Mesh: the debater

Structural components include the control of LLM calls and the collections of data artifacts across the topology.


'''

from dataclasses import dataclass
import json 
import time
from inference.agent_wiring import Agent, majority_state



@dataclass
class StepRecord:
    '''
    Stores per-step metadata for topology run.

    Data Include:
        - Proposals: First pass of the individual agents solving a particular step (i.e. agents 1-n solving step 2)
            - Chain: [single output]
            - Star: [leaf0, leaf1, ..., leaf-n] no judge
            - Mesh: [pass1_a0, pass1_a1, ..., pass1_a-n] 
        - Second Proposals: Only for the mesh, empty otherwise. Every agent recieves every other agents opinions
        - Final State: Concensus state that is brought to the next step 
        - Ground Truth: Correct state at the particular step

    '''
    step_index:   int
    topology:     str
    proposals:    list[dict]
    second_proposals: list[dict]
    final_state:  dict
    ground_truth: dict
    elapsed_sec:  float


@dataclass
class TopologyResult:
    '''
    Stores the meta-data for the overall topology run.
    '''
    topology:      str
    final_state:   dict
    step_records:  list[StepRecord]
    total_elapsed: float


## ----------
## CHAIN 
## ----------

def chain(agent: Agent, prompts, init_state):
    '''
    A sequential pipeline. Each agent only sees the output of the previous.

    Agent count is the length of the task. Each prompt is one agents full task.

    Agent 1 --> Agent 2 --> Agent N

    '''
    current_state = dict(init_state)
    step_records = []
    t_total_start = time.perf_counter()
    n_steps  = len(prompts)

    print(f"[TOPOLOGY] Starting Chain ({n_steps} steps)")

    #Creates agent chain based on prompt count
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()

        print(f"Processing Agent Step {i+1}/{n_steps}...", end="", flush=True)

        if i == 0:
            print(f"\n[DEBUG] Chain is Using: {prompt[:300]}...")

        #Context reset before the next agent inference occurs
        agent.context_reset()
        new_state = agent.execute_task(task=prompt, context=current_state)

        #Log to check for completion
        if isinstance(new_state, dict):
            val = new_state.get('Value', 'Missing')
            print(f"Done. (Value: {val})", flush=True)
        else:
            print("FAILED (Invalid/Empty JSON).", flush=True)


        elapsed = time.perf_counter() - t0

        #All step artifacts collected for the specific index of the loop
        step_records.append(StepRecord(
            step_index=i,
            topology='chain',
            proposals=[new_state],
            second_proposals=[],
            final_state=new_state, #since there its only from the previous agent
            ground_truth={},
            elapsed_sec=elapsed
        ))

        #the final state determined by the agent then becomes the input for the next agent
        current_state = new_state

    #After looping through all prompts the last agent output becomes the "final-final" state
    return TopologyResult(
        topology='chain',
        final_state=current_state,
        step_records=step_records,
        total_elapsed=time.perf_counter() - t_total_start
    )


## ----------
## STAR
## ----------

def star(agent: Agent, prompts, init_state, n_leaf_agents=5, rules_text=None):
    '''
    A node to center pipeline. The center sees all node inputs, but the nodes don't see each other.

                Agent 2
                   |
                   V
    Agent 1 --> Judge <-- Agent 3

    '''
    judge_state = dict(init_state)
    step_records = []
    t_total_start = time.perf_counter()

    #Ensures the number of steps is equal to the count of prompts
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()

        #Batches parallel inference for efficiency
        batch_tasks = [prompt] * n_leaf_agents
        batch_contexts = [judge_state] * n_leaf_agents

        if i == 0:
            print(f"\n[DEBUG] Judge is Using: {rules_text[:300]}...")

        #Context reset before the next node agent inference occurs
        agent.context_reset()

        #First independent node inference
        proposals = agent.execute_batch(tasks=batch_tasks, contexts=batch_contexts)

        #-- JUDGE PASS --

        #Separate prompt to ensure judge properly handles the node responses properly
        judge_task = (
            f'''
            You are a judge agent responsible for determining the single correct JSON state.

            You have received proposals from {n_leaf_agents} independent agents who each applied 
            the same ruleset to the same input state.

            RULESET TO FOLLOW:
            {rules_text}

            PROPOSALS FROM LEAF AGENTS:
            {json.dumps(proposals, indent=2)}

            CURRENT INPUT STATE (what all leaf agents received):
            {json.dumps(judge_state, indent=2)}

            Review the proposals critically and calculate your own response if you feel there are errors in the responses.
            Respond with ONLY a valid JSON object. No explanation, no extra text.
            Format: {{"Value": <float>, "Parity": "<A or B>", "Level": <integer>}}
            '''
        )

        #Context reset before the judge state receives the context
        agent.context_reset()

        #Judge inference with node agent context
        new_judge_state = agent.execute_task(task=judge_task, context=proposals)
        
        elapsed = time.perf_counter() - t0

        #All step artifacts collected for the specific index of the loop
        step_records.append(StepRecord(
            step_index=i,
            topology='star',
            proposals=proposals, #leaf agent responses
            second_proposals=[],
            final_state=new_judge_state, #judge determination at the step
            ground_truth={},
            elapsed_sec=elapsed
        ))

        #Goes all the way back to give the new nodes the judge's answer for the next step
        judge_state = new_judge_state

    #After looping through all prompts and node-judge passes the last judge output becomes the "final-final" state
    return TopologyResult(
        topology='star',
        final_state=judge_state,
        step_records=step_records,
        total_elapsed=time.perf_counter() - t_total_start
    )


## ----------
## MESH
## ----------

def mesh(agent: Agent, prompts, init_state, n_agents=5, rules_text=None):
    '''
    An agent-to-agent deliberation process.

    In the first pass each agent determines their own answers. 
    In the second pass, the agents determine their final answer based on others input.
    A majority vote is then used to determine the step-final answer from the agents 2-pass deliberation. 
    
    '''
    j_state = dict(init_state)
    step_records = []
    t_total_start = time.perf_counter()

    #Ensures the number of steps is equal to the count of prompts
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()

        #-- PASS 1: Independent Analysis --

        #Batches parallel inference for efficiency
        batch_tasks_1 = [prompt] * n_agents
        batch_contexts_1 = [j_state] * n_agents

        if i == 0:
            print(f"\n[DEBUG] Mesh is Using: {rules_text[:300]}...")
        
        #Ensures context is reset before the first inference round
        agent.context_reset()

        #First independent agent inference pass
        first_pass = agent.execute_batch(tasks=batch_tasks_1, contexts=batch_contexts_1)

        #-- PASS 2: Agent Deliberation --

        #Separate prompt to ensure debater properly handles the other agents responses properly
        deliberation_task = (
            f'''
            You are a reasoning agent in a peer deliberation round.

            In the first round, you and {n_agents - 1} other agents independently applied the same 
            ruleset to the same input state.

            RULESET TO FOLLOW:
            {rules_text}

            INPUT STATE (what all agents received in round one):
            {json.dumps(j_state, indent=2)}

            FIRST-ROUND PROPOSALS FROM ALL AGENTS:
            {json.dumps(first_pass, indent=2)}

            Review the proposals critically and calculate your own response if you feel there are errors in the responses.
            Respond with ONLY a valid JSON object. No explanation, no extra text.
            Format: {{"Value": <float>, "Parity": "<A or B>", "Level": <integer>}}
            '''
        )

        #Batches parallel inference for efficiency
        batch_tasks_2 = [deliberation_task] * n_agents
        batch_contexts_2 = [None] * n_agents

        #Ensures context is reset before the second inference round
        agent.context_reset()

        #Second debate style agent inference pass
        second_pass = agent.execute_batch(tasks=batch_tasks_2, contexts=batch_contexts_2)

        
        #-- MAJORITY VOTE --

        #Majority vote of the second inference step taken
        concensus = majority_state(second_pass)
        elapsed = time.perf_counter() - t0

        #All step artifacts collected for the specific index of the loop
        step_records.append(StepRecord(
            step_index=i,
            topology='mesh',
            proposals=first_pass, #individual analysis responses
            second_proposals=second_pass, #deliberation responses
            final_state=concensus, #the majority vote at the end of each step 
            ground_truth={},
            elapsed_sec=elapsed
        ))

        #The majority vote is then passed all the way up to the next round of individual analyses
        j_state = concensus

    #After looping through all prompts and agent-review passes the last majority vote output becomes the "final-final" state
    return TopologyResult(
        topology='mesh',
        final_state=j_state,
        step_records=step_records,
        total_elapsed=time.perf_counter() - t_total_start
    )

