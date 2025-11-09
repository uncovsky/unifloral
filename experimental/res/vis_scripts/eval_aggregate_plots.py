from rliable import library as rly
from rliable import metrics
import matplotlib.pyplot as plt
from rliable import plot_utils
import pandas as pd
import numpy as np
import seaborn as sns
from evaluation import load_results_dataframe
from tex_setup import set_size

"""
    Generates aggregate performance profiles
    and violin plots for final evaluation
"""

# Fetch results
df = load_results_dataframe("vis_data/unifloral_eval")
algorithms = sorted(df['algorithm'].unique())
datasets = sorted(df['dataset'].unique())
result_dict = {}


NUM_RUNS = 10  # expected runs per dataset

# Fit into expected format for rliable
for algorithm in algorithms:
    results = np.zeros((NUM_RUNS, len(datasets)))

    for j, dataset in enumerate(datasets):
        # filter rows
        algo_data = df[(df['algorithm'] == algorithm) &
                       (df['dataset'] == dataset)]
        print(algo_data)

        # extract scores
        scores = algo_data['final_scores_mean'].values.tolist()

        # pad or trim to fixed run count
        scores = (scores + [0.0] * NUM_RUNS)[:NUM_RUNS]

        results[:, j] = scores

    result_dict[algorithm] = results

thresholds = np.linspace(0.0, 100.0, 101)
score_distributions, score_cis = rly.create_performance_profile(
    result_dict, thresholds
)



colors = {algo: plt.cm.tab10(i) for i, algo in enumerate(algorithms)}
fig, ax = plt.subplots(figsize=set_size(width_fraction=1.0, height_fraction=0.3))
plot_utils.plot_performance_profiles(
  score_distributions, thresholds,
  performance_profile_cis=score_cis,
  colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
  xlabel=r'D4RL Normalized Score $(\tau)$',
  ax=ax)
plt.title('Performance Profile Across Datasets')


from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=colors[algorithm], lw=6, 
                label=algorithm.replace('_', ' ').upper(),
                markersize=12)  #Increased line width for larger appearance
           for algorithm in algorithms]

legend = fig.legend(
    handles=handles,
    labels=algorithms,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.05),  #position above the plot
    ncol=len(algorithms),
    frameon=True,  #enable border
    handlelength=0.5,
    fancybox=True,  #rounded corners
    framealpha=0.9, #slightly transparent
    borderpad=0.75,
    edgecolor='black',  #border color
    fontsize=8,
)

plt.tight_layout()
plt.subplots_adjust(top=0.85)  

plt.savefig("figures/perf.pdf", dpi=300, bbox_inches='tight')




from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

records = []
for algo, arr in result_dict.items():
    for run_idx in range(arr.shape[0]):
        for ds_idx in range(arr.shape[1]):
            records.append({
                "algorithm": algo,
                "score": min(arr[run_idx, ds_idx], 100.0)  # cap at 100
            })

df_long = pd.DataFrame(records)

algos = df_long["algorithm"].unique()
palette = sns.color_palette("tab10", len(algos))

fig, ax = plt.subplots(figsize=set_size(width_fraction=1.0,
                                        height_fraction=0.3))
sns.violinplot(
    data=df_long,
    x="algorithm",
    cut=0,
    scale='width',
    linewidth=1,
    color='#DDDDDD',
    saturation=1,
    order=algos,
    inner=None,
    y="score")

sns.stripplot(
    data=df_long,
    x="algorithm",
    y="score",
    jitter=True,
    order=algos,
    hue="algorithm",
    palette='tab10',      
    edgecolor='black',
    ax=ax,
    size=2)

_ = plt.xticks(rotation=45, ha='right')

ax.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.5)
ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.2, linewidth=0.3)

plt.ylabel("D4RL Normalized Score")
plt.xlabel("")

plt.legend().remove()
plt.savefig("figures/violin_basic.pdf", dpi=300, bbox_inches='tight')


