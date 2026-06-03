
'''

CLI script for executing topology experiments.

After activating venv, from the repo root: 

    **Local wiring test, mock model:**

    python rnnr_main.py --mock --n_trials 3 --n_workers 1

    
    **Tiny model test:**

    python rnnr_main.py --topology chain --light_model TinyLlama


    **Single topology, mid model:**

    python run_experiment.py --heavy_model Qwen/Qwen2.5-7B-Instruct

    
    **Single topology, heavy model:**

    python run_experiment.py --heavy_model Qwen/Qwen3-8B

    
    **Full experiment, all topologies, heavy model, 35 bootstrap trials:**

    python rnnr_main.py --topology all --mid_model Qwen/Qwen2.5-7B-Instruct --task_difficulty enhanced --temp 0.8 --top_p 0.5 --epsilon 15.0 --device cuda --n_trials 35 --n_workers 1 --enforce_eager --max_tokens 250 --out qwen25_full_spectral_run_t08_p05_e15_tr35.json



'''

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


from inference.experiment_rnnr import (
    run_bootstrap,
    run_all_topologies,
    save_results
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Run spectral multi-agent topology experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    #Below are for model specification

    p.add_argument(
        '--light_model',
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='HuggingFace model ID. Ignore if --mock is set'
    )

    p.add_argument(
        '--mid_model',
        default='Qwen/Qwen2.5-7B-Instruct',
        help='HuggingFace model ID for Qwen 2.5. Ignore if --mock is set'
    )

    p.add_argument(
        '--heavy_model',
        default='Qwen/Qwen3-8B',
        help='HuggingFace model ID. Ignore if --mock is set'
    )

    p.add_argument(
        '--device',
        default='cuda',
        help="PyTorch device string: 'cuda', 'cpu'."
    )

    #Below are for topology specification

    p.add_argument(
        '--topology',
        choices=['chain','star','mesh','all'],
        default='all',
        help="Which topology to run. 'all' runs chain, start, and mesh"
    )

    p.add_argument(
        '--temp',
        type=float,
        default=0.7,
        help='Sampling temperature for variance.'
    )

    p.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Nucleus sampling top_p threshold.'
    )

    #Below are the experiment params
    p.add_argument(
        '--n_trials',
        type=int,
        default=30,
        help='Bootstrap trial count.'
    )

    p.add_argument(
        '--n_leaf',
        type=int,
        default=4,
        help='Leaf agent count.'
    )

    p.add_argument(
        '--n_mesh',
        type=int,
        default=4,
        help='Mesh agent count.'
    )

    #Below are for setting the computational configs

    p.add_argument(
        '--n_workers',
        type=int,
        default=1,
        help='Parallel worker processes.'
    )

    p.add_argument('--max_tokens', 
                   type=int, 
                   default=128, 
                   help='Max tokens for generation'
    )

    p.add_argument('--disable_guided_json', 
                   action='store_true', 
                   help='Turn off JSON schema forcing'
    )

    p.add_argument(
        '--perturb_mode',
        choices=['numeric','noise_field'],
        default='numeric',
        help='Perturbation type for condition number measurement.'
    )

    p.add_argument(
        '--epsilon',
        type=float,
        default=0.5,
        help='Numeric perturbation amount for testing condition number.'
    )

    p.add_argument(
        '--value_key',
        default='Value',
        help='JSON used for numeric metric comparisons.'
    )

    p.add_argument(
        '--enforce_eager',
        action='store_true',
        help='Disable CUDA graphs to save memory.'
    )

    p.add_argument(
        '--task_difficulty',
        choices=['default', 'enhanced'],
        default='default',
        help='Toggles the prompt ruleset and ground truth data.'
    )

    #Below are for testing codebase functionality
    p.add_argument(
        '--mock',
        action='store_true',
        help='Uses MockAgent. No model or GPU required. For wiring tests.'
    )

    p.add_argument(
        '--noise_prob',
        type=float,
        default=0.0,
        help='MockAgent noise probability. 0.0 = clean run.'
    )

    #Below are for saving the results

    p.add_argument(
        '--data_dir',
        default='data',
        help='Directory to save results'
    )

    p.add_argument(
        '--out',
        default='results.json',
        help='Output filename written to data!'
    )

    return p.parse_args()


from inference.agent_wiring import Agent

def main():
    args = parse_args()

    #Code below routes pre-set args to the necessary backend components 
    selected_model = args.light_model 
    if '--heavy_model' in sys.argv:
        selected_model = args.heavy_model
    elif '--mid_model' in sys.argv:
        selected_model = args.mid_model

    print("\n" + "=" * 60)
    print("SPECTRAL MA TOPOLOGY EXPERIMENT")
    print("="*60)
    print(f"Topology: {args.topology}")
    print(f"Model: {selected_model}")
    print(f"Device: {args.device}")
    print(f"N-Trials: {args.n_trials}")
    #print(f"N-Agents: {args.n_agents}")
    print(f"N-Leaf: {args.n_leaf}")
    print(f"N-Mesh: {args.n_mesh}")
    print(f"N-Workers: {args.n_workers}")
    print(f"Perturb: {args.perturb_mode}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Temperature: {args.temp}")
    print(f"Top-P: {args.top_p}")
    print(f"Output: {args.out}")
    print("=" * 60 + "\n")

    t_start = time.perf_counter()

    #Adding shared agent to not re-load each instance of the model at each trial
    shared_agent = Agent(
        mock=args.mock,
        model_name=selected_model,
        device=args.device,
        noise_prob=args.noise_prob,
        enforce_eager=args.enforce_eager,
        max_tokens= args.max_tokens,
        use_guided_json= not args.disable_guided_json,
        temperature=args.temp,
        top_p=args.top_p
    )

    shared_kwargs = dict(
        agent=shared_agent,
        n_trials=args.n_trials,
        mock=args.mock,
        #model_name=selected_model,
        #device=args.device,
        #n_agents=args.n_agents,
        n_leaf=args.n_leaf,
        n_mesh=args.n_mesh,
        noise_prob=args.noise_prob,
        perturb_mode=args.perturb_mode,
        epsilon=args.epsilon,
        value_key=args.value_key,
        n_workers=args.n_workers,
        #enforce_eager=args.enforce_eager
        task_difficulty=args.task_difficulty
    )

    if args.topology == 'all':
        results = run_all_topologies(**shared_kwargs)
    else:
        results = {
            args.topology: run_bootstrap(topology=args.topology, **shared_kwargs)
        }

    out_path = save_results(results, filename=args.out, data_dir=args.data_dir)

    elpased = time.perf_counter() - t_start
    print(f'Experiment done! Total time {elpased}s')
    print(f'Results written to: {out_path}')

    #Adding a quick QA check of the file contents 
    if results and args.topology in results:
        sample_trial = results[args.topology][0]
        print("\n" + "-"*40)
        print(f"REAL-TIME QA CHECK:")
        print(sample_trial.get('baseline_final', 'No baseline state found'))
        print("-" * 40 + "\n")



if __name__=='__main__':
    main()



