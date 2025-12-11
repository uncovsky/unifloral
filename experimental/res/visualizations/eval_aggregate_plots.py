from rliable import library as rly
from rliable import metrics
import matplotlib.pyplot as plt
import re
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
df = load_results_dataframe("filter_eval")
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
        # extract scores
        scores = algo_data['final_scores_mean'].values.tolist()

        # pad or trim to fixed run count
        scores = (scores + [0.0] * NUM_RUNS)[:NUM_RUNS]

        results[:, j] = scores

    result_dict[algorithm] = results



from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_violin_by_dataset_groups(df_long, dataset_group, result_path="figures", 
                                  width_fraction=1.0, height_fraction=0.3,
                                  group_idx=0,
                                  algorithms=None):
    df_group = df_long[df_long["dataset"].isin(dataset_group)].copy()
    n_datasets = len(dataset_group)

    if algorithms is not None:
        df_group = df_group[df_group["algorithm"].isin(algorithms)]


    fig, axes = plt.subplots(1, n_datasets, figsize=set_size(width_fraction), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, dataset_group):
        df_ds = df_group[df_group["dataset"] == dataset]
        algos = sorted(df_ds["algorithm"].unique())

        sns.violinplot(
            data=df_ds,
            x="algorithm",
            y="score",
            cut=0,
            inner=None,
            bw=0.2,
            scale='width',
            linewidth=1.0,           # thin line
            palette=["#DDDDDD"] * len(algos),  # light gray fill
            edgecolor='black',
            saturation=0.5,
            alpha=0.3,
            order=algos,
            ax=ax
        )

        sns.stripplot(
            data=df_ds,
            x="algorithm",
            y="score",
            jitter=True,
            order=algos,
            palette='tab10',
            edgecolor='black',
            size=3,
            ax=ax
        )

        ax.set_title(dataset)
        # Transform labels: replace '_' with '-' and uppercase
        labels = [label.get_text().replace("_", "-").upper() for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.2, linewidth=0.3)

    axes[0].set_ylabel("D4RL Normalized Score")
    for ax in axes:
        ax.set_xlabel("")
    for ax in axes[1:]:
        ax.set_ylabel("")

    # Set common x-label in the middle
    fig.text(0.5, 0.02, "Algorithm", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at the bottom for the label
    plt.savefig(f"{result_path}/violin_group_{group_idx}.pdf", dpi=300)
    plt.close(fig)

def plot_violin_aggregate(df_long, result_path="figures", width_fraction=1.0, height_fraction=0.4,
                          datasets=None,
                          ext="",
                          algorithms=None, title="Aggregate Performance"):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    os.makedirs(result_path, exist_ok=True)

    df_plot = df_long.copy()
    if algorithms is not None:
        df_plot = df_plot[df_plot["algorithm"].isin(algorithms)]
    
    if datasets is not None:
        df_plot = df_plot[df_plot["dataset"].isin(datasets)]

    algos = sorted(df_plot["algorithm"].unique())

    
    fig, ax = plt.subplots(figsize=set_size(width_fraction, height_fraction))
    
    # Transparent gray violins
    palette = {algo: (0.7, 0.7, 0.7, 0.3) for algo in algos}  # RGBA
    
    sns.violinplot(
        data=df_plot,
        x="algorithm",
        y="score",
        inner=None,           # no median/box lines
        scale='width',
        bw=0.2,
        linewidth=0.8,
        alpha=0.3,
        palette=palette,
        order=algos,
        ax=ax
    )
    
 
    def clean_name(s):
        if "hopper" in s.lower() or "walker2d" in s.lower() or "halfcheetah" in s.lower():
            return s.split("-")[0]      # take first word
        else:
            return re.sub(r"-v\d+$", "", s)  # remove -vX suffix

    datasets_clean = [clean_name(ds) for ds in sorted(df_plot["dataset"].unique())]

    # Fixed RGB palette on cleaned names
    # Red → purple → blue gradient palette
    nice_palette = {
        ds: c for ds, c in zip(
            datasets_clean,
            ["#d7191c", "#7570b3", "#2b83ba"]
        )
    }

    # Apply cleaned names as a new column for plotting
    df_plot["dataset_clean"] = df_plot["dataset"].apply(clean_name)

    sns.stripplot(
        data=df_plot,
        x="algorithm",
        y="score",
        hue="dataset_clean",       # color by dataset
        jitter=True,
        order=algos,
        palette=nice_palette,
        edgecolor='black',
        size=2,
        dodge=True,         # keep all points on same x location
        ax=ax
    )

    # Legend BELOW the plot (centered)
    leg = ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),   # below plot
        ncol=len(nice_palette),         # single row 
        frameon=False,
        handlelength=0.6,       # tiny color markers
        handletextpad=0.1,      # extremely small gap between marker + text
        borderpad=0.05,         # tiny padding inside legend box
        labelspacing=0.2,      # almost no vertical space
        columnspacing=1.0,      # <<< controls spacing between legend entries
        fontsize=6
    )


        # Make legend text small
    for text in leg.get_texts():
        text.set_fontsize(7)

    # Make legend markers small
    for handle in leg.legendHandles:
        handle.set_markersize(5)


    
    # Transform x-labels: replace '_' with '-' and uppercase
    labels = [algo.replace("_", "-").upper() for algo in algos]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ax.set_ylabel("D4RL Normalized Score", fontsize=8)
    ax.set_xlabel("")
    ax.set_title(title)
    
    ax.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.2, linewidth=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{result_path}/violin_aggregate_{ext}.pdf", dpi=300)
    plt.close(fig)


mujoco_datasets = ['halfcheetah-medium-expert-v2', 'hopper-medium-v2',
                   'walker2d-medium-replay-v2']

adroit_datasets2 = ['pen-human-v1', 'pen-cloned-v1', 'pen-expert-v1',
                   'kitchen-mixed-v0']
adroit_datasets = ['pen-human-v1', 'pen-cloned-v1', 'kitchen-mixed-v0']

maze_datasets = ['antmaze-medium-diverse-v2']

dataset_groups = [mujoco_datasets, adroit_datasets, maze_datasets]

df_long = pd.melt(
    df,
    id_vars=['algorithm', 'dataset'],
    value_vars=['final_scores_mean'],
    var_name='metric',
    value_name='score'
)
# astype
df_long['score'] = df_long['score'].astype(float)
df_long['dataset'] = df_long['dataset'].astype(str)
df_long['algorithm'] = df_long['algorithm'].astype(str)


#algorithms = ['cql', 'edac', 'msg', 'pbrl', 'rebrac', 'sac_n']


#plot_violin_by_dataset_groups(df_long, adroit_datasets, algorithms=algorithms,
#                              width_fraction=0.9, height_fraction=0.1,
                              #group_idx=0)

plot_violin_aggregate(df_long, algorithms=algorithms,
                      width_fraction=0.5, height_fraction=0.25,
                      datasets=adroit_datasets,
                      ext="filter_adroit_franka",
                      title="")

plot_violin_aggregate(df_long, algorithms=algorithms,
                      width_fraction=0.5, height_fraction=0.25,
                      datasets=['pen-expert-v1'],
                      ext="filter_adroit_expert",
                      title="")

plot_violin_aggregate(df_long, algorithms=algorithms,
                      width_fraction=0.5, height_fraction=0.25,
                      datasets=mujoco_datasets,
                      ext="filter_mujoco",
                      title="")

plot_violin_aggregate(df_long, algorithms=algorithms,
                      width_fraction=0.5, height_fraction=0.25,
                      datasets=maze_datasets,
                      ext="filter_maze",
                      title="")
