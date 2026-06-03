
'''
Experimental Validation and Analysis

To evaluate whether empirical topology performance aligns with spectral-theoretic predictions

Produces:
- 1.) Three pillar table with theoretical, empirical, and consistency response. 
- 2.) Kruskal-Wallis H-test of the pillars
- 3.) Spearman rank agreement (descriptive only)


'''

import sys
import json
import warnings
import itertools
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt
import seaborn as sns

from src.spectral.spanalysis import Spectral, chain_adjacency, star_adjacency, mesh_adjacency

#For spectral analysis
GAMMA    = 0.9
#For chain/prompt steps
N_STEPS  = 12
#For star/leaf and mesh agent count
N_LEAVES = 4 

#For statistical testing and other loops...standard ordering
TOPO_ORDER = ["chain", "mesh", "star"]

#Predicted theoretical ranking for each pillar
PREDICTED_ORDER = {
    #Ordering is best - middle - worst
    "stability":   ("chain", "mesh", "star"),
    "convergence": ("mesh", "star", "chain"),
    "robustness":  ("chain", "mesh", "star")
}

#For specific ordering direction when determining rank
PILLAR_DIRECTION = {
    "stability":   "asc",
    "convergence": "asc",
    "robustness":  "asc",
}


#Maps pillar names to theoretical features
PILLAR_THEORETICAL_KEY = {
    "stability":   "spectral_radius",
    "convergence": "spectral_gap",
    "robustness":  "condition_number"
}


TOPO_COLORS = {
    "chain":"#8B0015", 
    "mesh":"#378DBD",
    "star":"#001C48"
}


## -------------------
## Loading and Extract
## -------------------

def load_results(path):
    '''
    Loading raw experiment JSON.

    '''
    with open(path) as f:
        return json.load(f)


def extract_scores(results):
    '''
    Converts JSON trials to pillar-aligned arrays.

    Returns a JSON structure of { "chain": {"stability": array, "convergence": array, "robustness": array}}

    Missing vals stored as NaN
    '''
    key_map = {
        'stability':'spectral_radius',
        'convergence':'spectral_gap',
        'robustness':'condition_number'
    }
    out={}
    #Loops through topologies and trials
    for topo, trials in results.items():
        arrays = {pillar: [] for pillar in key_map}
        #In each topology iterate through the trials for the eval score 
        for trial in trials:
            e = trial.get('eval', {})
            #Within each of the trials obtain the various scores and append the name mapping
            for pillar, eval_key in key_map.items():
                block = e.get(eval_key)
                val = block['score'] if (block and block['score'] is not None) else float('nan')
                arrays[pillar].append(val)
        #Adds values to a formal array
        out[topo] = {p: np.array(v) for p, v in arrays.items()}
    return out




## -------------------
## Spectral Features
## -------------------

#This will be done formally before the table, but still is useful for quick comparison 
def compute_spectral_features():
    '''
    Computes the theoretical spectral metrics for each topology.
    
    '''
    #Creates the spectral analysis features
    specs = {
        'chain': Spectral(chain_adjacency(N_STEPS), gamma=GAMMA),
        'star': Spectral(star_adjacency(N_LEAVES), gamma=GAMMA),
        'mesh': Spectral(mesh_adjacency(N_LEAVES), gamma=GAMMA)
    }
    return {
        #Obtains the spectral measurements
        name: {
            'spectral_radius': sp.full_single_analysis().spectral_radius,
            'spectral_gap': sp.full_single_analysis().spectral_gap,
            'condition_number': sp.full_single_analysis().condition_number
        }
        for name, sp in specs.items()
    }




## --------------------
## Ordering Consistency
## --------------------

def ordering_consistency(scores, pillar):
    '''
    **[DEVELOPMENT ONLY]**

    For each trial, checks whether the empirical matches the predicted rank ordering.

    Returns the % of trials where the full predicted ordering holds. 

    '''
    best, mid, worst = PREDICTED_ORDER[pillar]

    #Align to the shortest topology trial count so all can be compared index-by-index
    n = min(len(scores[t][pillar]) for t in [best, mid, worst])
    b = scores[best][pillar][:n]
    m = scores[mid][pillar][:n]
    w = scores[worst][pillar][:n]

    #If not null then valid
    valid = ~(np.isnan(b) | np.isnan(m) | np.isnan(w))
    b, m, w = b[valid], m[valid], w[valid]

    #If all are invalid return safely
    if len(b) == 0:
        return float('nan'), 0
    
    matches = np.sum((b < m) & (m < w))
    return round(100 * matches / len(b), 1), len(b)



## --------------------
## Statistical Tests
## --------------------

def run_kruskal(scores, pillar):
    '''
    Kruskal-Wallis H-test across the topology trial distributions for the empirical metrics.

    Null hypothesis:
    - all topology score distribution medians are the same

    Alternative:
    - at least one topology distribution median differs
    
    '''
    groups = []
    #Loops through each topology to obtain the scores by pillar for building a distribution
    for topo in TOPO_ORDER:
        v = scores[topo][pillar]
        groups.append(v[~np.isnan(v)])
    #Obtains the H stats and p-val
    H, p = kruskal(*groups)
    return {'H':H, 'p':p}



def run_spearman(scores, pillar):
    '''
    Runs a Spearman rank correlation between the theoretical and empirical values across the three dfferent topologies.

    Given low number of topologies this is NOT inferential and used to check ordinal agreement.
    
    '''
    pred_best, pred_mid, pred_worst = PREDICTED_ORDER[pillar]

    #Convert predicted ordering into ordinal ranks
    theoretical_rank = {pred_best: 1, pred_mid: 2, pred_worst: 3}

    #Calculates the mean of the trial scores by pillar for each topology
    means = {topo: np.nanmean(scores[topo][pillar]) for topo in TOPO_ORDER}

    #Sorts topologies by their empirical mean
    sorted_topos = sorted(TOPO_ORDER, key=lambda t: means[t])

    #Adds count value to the ordered empirical means scores to assign ordinal structure
    empirical_rank = {topo: i+1 for i, topo in enumerate(sorted_topos)}

    #Converts rank dictionaries into aligned rank lists
    t_ranks = [theoretical_rank[t] for t in TOPO_ORDER]
    e_ranks = [empirical_rank[t] for t in TOPO_ORDER]

    #Runs correlation to obtain just the coefficient
    rs, _ = spearmanr(t_ranks, e_ranks)
    return {'rs': rs, 'theoretical_rank': theoretical_rank, 'empirical_rank': empirical_rank, 'means': means}




## --------------------
## Table Builder
## --------------------

def build_table(scores, spec_features):
    '''
    Builds a lightweight text summary table.

    Includes the: Theoretical | empirical mean+-sd | consistency %

    '''
    pillars = {
        'convergence': 'spectral_gap',
        'stability': 'spectral_radius',
        'robustness': 'condition_number'
    }

    
    #Computes consistency for each metric
    consistency = {
        pillar: ordering_consistency(scores, pillar)[0]
        for pillar in pillars
    }

    lines=[]
    sep = '-' * 130

    #Creates header for the table
    lines.append(sep)
    lines.append(
        f"{'Topology':<10} | {'Convergence':<26} | "
        f"{'Stability':<26} | {'Robustness':<26} | {'Failure Rate':<12}"
    )
    lines.append(sep)

    #Loops through topology by theoretically predicted order
    for topo in TOPO_ORDER:
        trial_proxy = scores[topo]['robustness']
        total_trials = len(trial_proxy)
        valid_trials = len(trial_proxy[~np.isnan(trial_proxy)])
        failure_rate = ((total_trials - valid_trials) / total_trials) * 100 if total_trials > 0 else 0.0

        row_parts = [f"{topo.capitalize():<10}"]

        #Within each topology the spectral predictions, empirical scores, and consistency obtained 
        for pillar, spec_key in pillars.items():
            theory = spec_features[topo][spec_key]

            emp = scores[topo][pillar]
            emp = emp[~np.isnan(emp)]

            mean = np.mean(emp) if len(emp) > 0 else float('nan')
            sd = np.std(emp) if len(emp) > 0 else float('nan')
            pct = consistency[pillar]

            row_parts.append(
                f"{theory:.3f} | {mean:.2f}±{sd:.2f} | {pct:.1f}%"
            )

        row_parts.append(f"{failure_rate:.1f}%")
        lines.append(" | ".join(row_parts))

    lines.append(sep)
    lines.append("Format: theoretical | empirical mean±sd | consistency % | failure %")

    return "\n".join(lines)



## --------------------
## Plot Builder
## --------------------

def setup_plot_style():
    '''
    Global setup of plot parameters.

    '''
    plt.rcParams.update({
        #"font.family": "serif",
        #"font.family": "times new roman",
        "font.family": "arial",
        #"axes.titlelocation": 'left',
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False
    })



def make_raincloud(scores, pillar):
    '''
    Generates a raincloud plot for a specific pillar of all three topologies.

    '''
    titles = {
        "stability": "Stability (Spectral Radius $\\rho$)",
        "convergence": "Convergence (Spectral Gap $\\Delta$)",
        "robustness": "Robustness (Condition Number $\\kappa$)"
    }
    ylabels = {
        "stability": "Cumulative Error",
        "convergence": "Consensus Score",
        "robustness": "Perturbation Sensitivity"
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(42)

    for i, topo in enumerate(TOPO_ORDER):
        vals = scores[topo][pillar]
        vals = vals[~np.isnan(vals)]
        color = TOPO_COLORS[topo]
        
        #Adding the individual data points
        jitter = rng.uniform(-0.05, 0.05, len(vals))
        ax.scatter(i - 0.2 + jitter, vals, color=color, alpha=0.4, s=15, zorder=3)
        
        #Adding the umbrella portion
        ax.boxplot(vals, positions=[i], widths=0.1, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color="black", linewidth=1.5),
                   flierprops=dict(marker=''))
        
        #Adding the cloud portion
        if len(vals) > 1:
            kde = gaussian_kde(vals)
            y_grid = np.linspace(vals.min(), vals.max(), 100)
            k = kde(y_grid)/kde(y_grid).max() * 0.3
            ax.fill_betweenx(y_grid, i + 0.1, i + 0.1 + k, color=color, alpha=0.3)


    ax.set_xticks(range(len(TOPO_ORDER)))
    ax.set_xticklabels([t.capitalize() for t in TOPO_ORDER])
    #Modify to change the casing of the text
    ax.set_title(titles.get(pillar, pillar.capitalize()), fontweight="bold")
    ax.set_ylabel(ylabels.get(pillar, "Value"))
    
    plt.tight_layout()
    plt.show()



def make_trace(results, topo, pillar_key):
    '''
    Generates a trace-plot for one topology for one pillar.

    Only error growth (spectral radius) and concensus (spectral gap) can be included since perturbation robustness is based on the final state only. 

    '''
    ylabels = {
        "spectral_radius": "Error Growth",
        "spectral_gap": "Consensus State Value"
    }
    
    fig, ax = plt.subplots(figsize=(9, 5))
    trials = results.get(topo, [])
    all_steps = []
    
    #Loops through the trials
    for t in trials:
        steps = t.get('eval', {}).get(pillar_key, {}).get('per_step', [])
        #If steps exists it plots the values
        if steps and len(steps) > 0:
            clean_steps = [float(x) if x is not None else np.nan for x in steps]
            
            ax.plot(range(len(clean_steps)), clean_steps, color='gray', alpha=0.2, linewidth=0.6)
            all_steps.append(clean_steps)
    
    #Plots the bold median line
    if all_steps:
        all_steps_array = np.array(all_steps, dtype=float)
        #mean_line = np.nanmean(all_steps_array, axis=0) <-- use if the mean is prefered
        median_line = np.nanmedian(all_steps_array, axis=0)
        ax.plot(range(len(median_line)), median_line, color=TOPO_COLORS[topo],
        #ax.plot(range(len(mean_line)), mean_line, color=TOPO_COLORS[topo], <-- use if the mean is prefered 
                linewidth=2.5, label=f"Median {topo.capitalize()}")
    
    #Use {pillar_key.replace('_', ' ').capitalize()} to keep capitalization
    ax.set_title(f"{topo.capitalize()} Empirical Stepwise Results - {pillar_key.replace('_', ' ').title()}", fontweight="bold")
    ax.set_xlabel("Reasoning Step Index")
    ax.set_ylabel(ylabels.get(pillar_key, "Value"))
    #ax.legend(frameon=False)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


## --------------------
## Print Helpers
## --------------------

def print_krusal_results(kw, pillar):
    '''
    Simple printout of KW H-test.
    
    '''
    sig = '**' if kw['p'] < 0.01 else '*' if kw['p'] < 0.05 else 'ns'
    print(f"\n{pillar.upper()}")
    print(f"Kruskal-Wallis: H={kw['H']:.3f}, p={kw['p']:.4f} [{sig}]")
    



def print_spearman(result, pillar):
    '''
    Simple printout of the Spearman rank agreement.
    
    '''
    #Theoretical rank
    theo = " | ".join(
        f"{t}:{result['theoretical_rank'][t]}"
        for t in TOPO_ORDER
    )

    #Empirical rank
    emp = " | ".join(
        f"{t}:{result['empirical_rank'][t]}"
        for t in TOPO_ORDER
    )

    print(f"\n{pillar.upper()}")
    print(f"Spearman rs = {result['rs']:+.3f} (n=3)")
    print(f"Theory: {theo}")
    print(f"Empirical: {emp}")




## --------------------
## Main
## --------------------

def main(results_path):
    '''
    Full experimental validation pipeline.

    Steps:
        - Loads JSON
        - Extracts empirical pillar scores
        - Computes theoretical spectral metrics
        - Print summary table
        - Run Kruskal tests 
        - Run Spearman rank checks
        - Compute ordering consistency
        - Returns reusable results dict
    
    '''
    pillars = {'stability','convergence','robustness'}

    results_path = Path(results_path)
    print(f"\nLoading: {results_path}")

    #Load - extract - and compute spectral 
    results = load_results(results_path)
    scores = extract_scores(results)
    spec_features = compute_spectral_features()

    #Creates general summary table
    print("\n" + build_table(scores, spec_features))

    #KW results for each of the spectral pillars 
    print("\nKRUSKAL-WALLIS")
    kruskal_results = {}
    for pillar in pillars:
        kw = run_kruskal(scores, pillar)
        kruskal_results[pillar] = kw
        print_krusal_results(kw, pillar)

    #Spearman rank agreement for each of the spectral pillars
    print("\nSPEARMAN RANK AGREEMENT")
    spearman_results = {}
    for pillar in pillars:
        sp = run_spearman(scores, pillar)
        spearman_results[pillar] = sp
        print_spearman(sp, pillar)

    #Consistency results
    consistency = {
        pillar: ordering_consistency(scores, pillar)
        for pillar in pillars
    }

    #Plotting results
    setup_plot_style()

    #stability, convergence, robustness
    pillarz = ['stability', 'convergence', 'robustness']
    for i in pillarz:
        make_raincloud(scores, i)


    topoz = ['chain','mesh','star']
    spectral_metricz = ['spectral_gap', 'spectral_radius']
    for i in topoz:
        for j in spectral_metricz:
            make_trace(results,i,j)
    

    return {
        "scores": scores,
        "spec_features": spec_features,
        "kruskal": kruskal_results,
        "spearman": spearman_results,
        "consistency": consistency
    }



if __name__=="__main__":
    main("data/results.json")





