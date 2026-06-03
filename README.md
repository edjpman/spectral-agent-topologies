# Predictive Maps of Multi-Agent Reasoning  
*A Successor-Representation Spectrum for LLM Communication Topologies*

## Project Overview

This repository contains the full experimental pipeline, analysis framework, and supporting materials for the study:

> **Predictive Maps of Multi-Agent Reasoning: A Successor-Representation Spectrum for LLM Communication Topologies**

The project investigates whether the structure of communication in multi-agent large language model (LLM) systems can predict reasoning behavior *before inference occurs*. Specifically, the work studies whether spectral properties of a successor representation (SR) constructed from the communication graph can predict:

- Cumulative Reasoning Drift
- Consensus Dynamics
- Robustness to Perturbations

Multi-agent LLM systems are modeled as directed communication graphs where agents exchange intermediate reasoning states. From the row-stochastic communication operator *P*, we construct the successor representation:


*M = (I - gamma P)^{-1}*


and analyze three spectral quantities:

- Spectral radius - *rho(M)*
- Spectral gap - *delta(M)*
- Condition number - *kappa(M)*

The study evaluates three communication topologies:

- Chain
- Star
- Mesh

using a structured 12-step state-tracking task executed with *Qwen2.5-7B-Instruct* over 100 independent trials per condition.

The experiments demonstrate that by topology:

- Condition number: perfectly predicts perturbation robustness
- Spectral gap: partially predicts consensus dynamics
- Spectral radius: inversely predicts cumulative error due to sequential reasoning drift


---

## Objectives & Research Questions

This project explores whether communication topology alone can predict reasoning behavior in multi-agent LLM systems prior to inference execution.

The primary research questions are:

1. Can multi-agent LLM systems be represented as predictive communication maps using successor representations?

2. Do spectral properties of the successor representation predict empirical reasoning behavior across different communication topologies?

3. Why can a topology that is spectrally stable still exhibit poor empirical reasoning performance?

4. Can topology selection become a pre-inference linear-algebraic problem rather than a purely empirical trial-and-error process?

The work is intended as a controlled case study and initial framework rather than a universal claim about all multi-agent systems.

---

## Experimental Task

Each trial performs a 12-step structured state-tracking task involving a JSON state with three coupled fields:

```json
{
  "Value": float,
  "Parity": "A or B",
  "Level": 1-9
}
```

The transition rules are intentionally order-dependent and combine:

- Arithmetic updates
- Conditional branching
- Logic dependencies

Agent contexts are reset between steps so that reasoning errors propagate only through the explicit communication topology.

Task execution and topology wiring are primarily implemented in:

```text
src/inference/experiment_rnnr.py
src/inference/agent_wiring.py
```

---

## Methodology

### Communication Graph Representation

Multi-agent systems are represented as directed graphs:


*G = (V, E)*


with row-normalized adjacency matrix *P*.

The successor representation is defined as:


*M = (I - gamma P)^{-1}*


with *gamma = 0.9*.


### Spectral Diagnostics

Three spectral quantities are extracted from \(M\):

| Quantity | Interpretation |
|---|---|
| Spectral Radius *rho(M)* | Predicts error amplification and drift tendency |
| Spectral Gap *Delta(M)* | Predicts consensus dynamics and mixing behavior |
| Condition Number *kappa(M)* | Predicts perturbation sensitivity and robustness |

### Empirical Metrics

Theoretical predictions are compared against empirical measurements:

- Cumulative Error Growth
- Consensus Decay Rate
- Perturbation Sensitivity

### Experimental Setup

- **Model:** Qwen2.5-7B-Instruct
- **Trials:** 100 per topology
- **Topologies:** Chain, Star, Mesh
- **Hardware:** NVIDIA A100 (32GB)
- **Decoding:** temperature = 0.8, top-p = 0.5
- **Perturbation magnitude:** epsilon = 15.0

---

## Repository Structure

The repository is organized into three primary components: experimental execution, empirical outputs, and analytical evaluation.

```text
root/
├── data/                     #Saved experiment outputs and JSON results
├── src/                      #Core experimental pipeline
│   ├── agents/               #Agent implementations and communication logic
│   ├── eval/           #Metrics and post-processing
│   └── spectral/             #Successor representation and spectral analysis tools
├── notebooks/
│   ├── spectral_sandbox.ipynb
│   └── experimental_pipeline.ipynb
├── README.md
├── requirements.txt
├── rnnr_main.py              #Main experiment entry point
├── hf_test.py
├── qwen3_8b_smoketest.py
├── test_agent_speed.py
└── vllm_smoketest.py
```

### Directory Summary

- **data/**  
  Stores empirical experiment outputs and processed JSON result files.

- **src/**  
  Contains the full backend experiment pipeline including topology construction, inference execution, evaluation metrics, and spectral analysis utilities.

- **notebooks/**  
  Jupyter notebooks for exploratory spectral analysis and empirical result evaluation.

- **Root-level scripts**  
  Utility and smoke-test files used for validating model loading, infrastructure setup, and execution performance.

---

## Setup

### Requirements

- Python 3.11.4
- GPU recommended for full experimentation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Setup

Experiments use:

```text
Qwen/Qwen2.5-7B-Instruct
```

The specific model commit used in experimentation is:

```text
a09a354
```

The model can be downloaded through Hugging Face CLI or another preferred method.

---

## Running the Experiment

To execute the full experiment pipeline:

```bash
python rnnr_main.py --topology all \
  --mid_model Qwen/Qwen2.5-7B-Instruct \
  --task_difficulty enhanced --temp 0.8 --top_p 0.5 \
  --epsilon 15.0 --device cuda --n_trials 100 --n_workers 1 \
  --enforce_eager --max_tokens 250 \
  --out qwen25_full_spectral_run_t08_p05_e15_tr100.json
```

Results will be saved to:

```text
/data/
```

---

## Key Findings

### Stability Paradox

The chain topology is the most spectrally stable configuration yet it produces the largest empirical cumulative error growth.

This behavior resembles a sequential “telephone game” effect where small reasoning biases accumulate monotonically across agents.

### Condition Number as a Predictive Diagnostic

The condition number perfectly predicts rank-order of topology perturbation robustness without requiring prior inference execution.

This suggests that spectral conditioning may serve as a practical zero-shot diagnostic for topology selection.

### Aggregation Suppresses Drift

Star and mesh topologies reduce cumulative reasoning drift through aggregation and deliberation mechanisms.

The experiments suggest that system resilience depends not only on the number of agents, but on how those agents are connected.

---

## Limitations and Future Work

This work is intentionally scoped as a controlled case study.

Current limitations include:

- Evaluation on only three communication topologies
- Use of a single model
- A single structured task
- Limited statistical power from topology-level rank comparisons

Future work includes:

- Evaluating richer and more diverse communication graphs
- Testing across additional reasoning tasks and models
- Extending the framework with broader algebraic and representational diagnostics

---

## Broader Impact

Structural pre-inference diagnostics may help practitioners design safer and more reliable multi-agent systems while reducing wasted compute from unstable communication structures.

At the same time, topology-level optimization tools could potentially be used to maximize adversarial perturbation amplification. Future work should therefore pair structural diagnostics with explicit adversarial robustness evaluation.

This project uses only synthetic task data and open-weight language models. No human subjects or personally identifiable information are involved.