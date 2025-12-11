from tex_setup import set_size

"""

    Runs UCB evaluation from Unifloral.
    - taken entirely from unifloral, except for data & visualization pipeline.


This module provides tools for:
1. Loading and parsing experiment results
2. Running bandit-based policy selection 
3. Computing confidence intervals via bootstrapping 



Evaluation:
        Subsample 5 policies from the trained 10 (500 times)
        Run bandit trial for 100 pulls, estimate best policy at each step
"""

from collections import namedtuple
from datetime import datetime
import os
import re
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd

import warnings

from functools import partial
import glob
import jax
from jax import numpy as jnp
import numpy as np



r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Data loading
"""


def parse_and_load_npz(filename: str) -> Dict:
    """Load data from a result file and parse metadata from filename.

    Args:
        filename: Path to the .npz result file

    Returns:
        Dictionary containing loaded arrays and metadata
    """
    # Parse filename to extract algorithm, dataset, and timestamp
    split = filename.split("/")

    if len(split) < 4:
        data = split[-1].split("_")
        algorithm = data[0]
        dataset = data[1]
        dt_str = data[2]

    else:
        dt_str = split[-3]
        algorithm = split[-4].split("_")[0]
        dataset = split[-4].split("_")[-1]

    data = np.load(filename, allow_pickle=True)
    data = {k: v for k, v in data.items()}
    data["algorithm"] = algorithm
    data["dataset"] = dataset
    data["datetime"] = dt_str
    data.update(data.pop("args", np.array({})).item())  # Flatten args

    if data["algorithm"] == "awac" and data["num_critics"] > 2:
        data["algorithm"] = f"awac_n"
    if data["algorithm"] == "pbrl" and data["critic_regularizer"] == "filtered_pbrl":
        data["algorithm"] = f"pbrl_f"
    return data


def load_results_dataframe(results_dir: str = "final_returns") -> pd.DataFrame:
    """Load all result files from a directory into a pandas DataFrame.

    Args:
        results_dir: Directory containing .npz result files

    Returns:
        DataFrame containing results from all successfully loaded files
    """
    npz_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))
    data_list = []

    for f in npz_files:
        try:
            data = parse_and_load_npz(f)
            data_list.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    df = pd.DataFrame(data_list).drop(columns=["Index"], errors="ignore")
    if "final_scores" in df.columns:
        df["final_scores"] = df["final_scores"].apply(lambda x: x.reshape(-1))
    if "final_returns" in df.columns:
        df["final_returns"] = df["final_returns"].apply(lambda x: x.reshape(-1))

    df = df.sort_values(by=["algorithm", "dataset", "datetime"])
    return df.reset_index(drop=True)


r"""
          __/)
       .-(__(=:
    |\ |    \)
    \ ||
     \||
      \|
    ___|_____
    \       /
     \     /
      \___/     Bandit Evaluation and Bootstrapping
"""

BanditState = namedtuple("BanditState", "rng counts rewards total_pulls")


def ucb(
    means: jnp.ndarray, counts: jnp.ndarray, total_counts: int, alpha: float
) -> jnp.ndarray:
    """Compute UCB exploration bonus.

    Args:
        means: Array of empirical means for each arm
        counts: Array of pull counts for each arm
        total_counts: Total number of pulls across all arms
        alpha: Exploration coefficient

    Returns:
        Array of UCB values for each arm
    """
    exploration = jnp.sqrt(alpha * jnp.log(total_counts) / (counts + 1e-9))
    return means + exploration


def argmax_with_random_tiebreaking(rng: jnp.ndarray, values: jnp.ndarray) -> int:
    """Select maximum value with random tiebreaking.

    Args:
        rng: JAX PRNGKey
        values: Array of values to select from

    Returns:
        Index of selected maximum value
    """
    mask = values == jnp.max(values)
    p = mask / (mask.sum() + 1e-9)
    return jax.random.choice(rng, jnp.arange(len(values)), p=p)


@partial(jax.jit, static_argnums=(2,))
def run_bandit(
    returns_array: jnp.ndarray,
    rng: jnp.ndarray,
    max_pulls: int,
    alpha: float,
    policy_idx: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run a single bandit algorithm and report results after each pull.

    Args:
        returns_array: Array of returns for each policy and rollout
        rng: JAX PRNGKey
        max_pulls: Maximum number of pulls to execute
        alpha: UCB exploration coefficient
        policy_idx: Indices of policies to consider

    Returns:
        Tuple of (pulls, estimated_bests)
    """
    returns_array = returns_array[policy_idx]
    num_policies, num_rollouts = returns_array.shape

    init_state = BanditState(
        rng=rng,
        counts=jnp.zeros(num_policies, dtype=jnp.int32),
        rewards=jnp.zeros(num_policies),
        total_pulls=1,
    )
    def bandit_step(state: BanditState, _):
        """Run one bandit step and track performance."""
        rng, rng_lever, rng_reward = jax.random.split(state.rng, 3)

        # Select arm using UCB
        means = state.rewards / jnp.maximum(state.counts, 1)
        ucb_values = ucb(means, state.counts, state.total_pulls, alpha)
        arm = argmax_with_random_tiebreaking(rng_lever, ucb_values)

        # Sample a reward for the chosen arm
        idx = jax.random.randint(rng_reward, shape=(), minval=0, maxval=num_rollouts)
        reward = returns_array[arm, idx]
        new_state = BanditState(
            rng=rng,
            counts=state.counts.at[arm].add(1),
            rewards=state.rewards.at[arm].add(reward),
            total_pulls=state.total_pulls + 1,
        )

        # Calculate best arm based on current state
        updated_means = new_state.rewards / jnp.maximum(new_state.counts, 1)
        best_arm = jnp.argmax(updated_means)
        estimated_best = returns_array[best_arm].mean()

        return new_state, (state.total_pulls, estimated_best)

    _, (pulls, estimated_bests) = jax.lax.scan(
        bandit_step, init_state, length=max_pulls
    )
    return pulls, estimated_bests


def run_bandit_trials(
    returns_array: jnp.ndarray,
    seed: int = 17,
    num_subsample: int = 20,
    num_repeats: int = 1000,
    max_pulls: int = 200,
    ucb_alpha: float = 2.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run multiple bandit trials and collect results at each step.

    Args:
        returns_array: Array of returns for each policy and rollout
        seed: Random seed
        num_subsample: Number of policies to subsample on each trial
        num_repeats: Number of trials to run
        max_pulls: Maximum number of pulls per trial
        ucb_alpha: UCB exploration coefficient

    Returns:
        Tuple of (pulls, estimated_bests)
    """
    rng = jax.random.PRNGKey(seed)
    num_policies = returns_array.shape[0]

    num_subsample = min(num_subsample, num_policies)
    if num_subsample > num_policies:
        warnings.warn("Not enough policies to subsample, using all policies")

    rng, rng_trials, rng_sample = jax.random.split(rng, 3)
    rng_trials = jax.random.split(rng_trials, num_repeats)

    def sample_policies(rng: jnp.ndarray) -> jnp.ndarray:
        """Sample a subset of policy indices."""
        if num_subsample > num_policies:
            return jnp.arange(num_policies)
        return jax.random.choice(
            rng, jnp.arange(num_policies), shape=(num_subsample,), replace=False
        )

    # Create a batch of policy index arrays for all trials
    rng_sample_keys = jax.random.split(rng_sample, num_repeats)
    policy_indices = jax.vmap(sample_policies)(rng_sample_keys)

    # Run bandit trials with policy subsampling
    # Pulls are the same for all trials, so we can just return the first one
    vmap_run_bandit = jax.vmap(run_bandit, in_axes=(None, 0, None, None, 0))
    pulls, estimated_bests = vmap_run_bandit(
        returns_array, rng_trials, max_pulls, ucb_alpha, policy_indices
    )
    return pulls[0], estimated_bests


def bootstrap_confidence_interval(
    rng: jnp.ndarray,
    data: jnp.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean of data.

    Args:
        rng: JAX PRNGKey
        data: Array of values to bootstrap
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level (between 0 and 1)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """

    @jax.vmap
    def bootstrap_mean(rng):
        samples = jax.random.choice(rng, data, shape=(data.shape[0],), replace=True)
        return samples.mean()

    bootstrap_means = bootstrap_mean(jax.random.split(rng, n_bootstraps))
    lower_bound = jnp.percentile(bootstrap_means, 100 * (1 - confidence) / 2)
    upper_bound = jnp.percentile(bootstrap_means, 100 * (1 + confidence) / 2)
    return lower_bound, upper_bound


def bootstrap_bandit_trials(
    returns_array: jnp.ndarray,
    seed: int = 17,
    num_subsample: int = 20,
    num_repeats: int = 1000,
    max_pulls: int = 200,
    ucb_alpha: float = 2.0,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, np.ndarray]:
    """Run bandit trials and compute bootstrap confidence intervals.

    Args:
        returns_array: Array of returns for each policy and rollout has shape (num_policies, num_rollouts)
        seed: Random seed
        num_subsample: Number of policies to subsample
        num_repeats: Number of bandit trials to run
        max_pulls: Maximum number of pulls per trial
        ucb_alpha: UCB exploration coefficient
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level for intervals

    Returns:
        Dictionary with the following keys:
        - pulls: Number of pulls at each step
        - estimated_bests_mean: Mean of the currently estimated best returns across trials
        - estimated_bests_ci_low: Lower confidence bound for estimated best returns
        - estimated_bests_ci_high: Upper confidence bound for estimated best returns
    """
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, max_pulls)

    pulls, estimated_bests = run_bandit_trials(
        returns_array, seed, num_subsample, num_repeats, max_pulls, ucb_alpha
    )

    # Estimated bests has shape (num_repeats, max_pulls)
    vmap_bootstrap = jax.vmap(bootstrap_confidence_interval, in_axes=(0, 1, None, None))

    # bootstrap CI for each pull step

    ci_low, ci_high = vmap_bootstrap(rng, estimated_bests, n_bootstraps, confidence)
    estimated_bests_mean = estimated_bests.mean(axis=0)

    return {
        "pulls": pulls,
        "estimated_bests_mean": estimated_bests_mean,
        "estimated_bests_ci_low": ci_low,
        "estimated_bests_ci_high": ci_high,
    }


if __name__ == "__main__":
    """
        generates the ucb evaluation plots for all algos/datasets

        switch between full eval / uwac eval / filtering eval
    """

    # Full evaluation data
    df = load_results_dataframe("full_eval_data")

    # PBRL filtering data
    #df = load_results_dataframe("filter_eval_data")

    # UWAC evaluation data (for ablation and full eval)
    # df = load_results_dataframe("uawac_eval_data")

    #fig, axes = plt.subplots(3, 3, figsize=set_size(width_fraction=0.5,
    #                                                height_fraction=0.5, subplots=(3, 3)), 
                             #sharex=True, sharey=False)
    fig_dims = set_size(width_fraction=1.0, subplots=(3, 3))
    fig, axes = plt.subplots(3, 3, figsize=fig_dims,
                                sharex=True, sharey=False)
    axes = axes.flatten()
    datasets = df["dataset"].unique()
    algorithms = df["algorithm"].unique()

    # Create consistent color mapping for algorithms
    colors = plt.cm.tab10.colors[:10]
    color_map = {alg: colors[i % len(colors)] for i, alg in enumerate(algorithms)}
    all_results = {}

    datasets = [
        "halfcheetah-medium-expert-v2",
        'hopper-medium-v2',
        'walker2d-medium-replay-v2',
        'pen-human-v1',
        'pen-cloned-v1',
        'pen-expert-v1',
        'antmaze-medium-diverse-v2',
        'maze2d-large-v1',
        'kitchen-mixed-v0',
    ]


    """
     Can further limit algos here, we use AWAC variants for critic ablation
        also fixed colors for plots

        awac ablation, don't show pbrl/rebrac
    algorithms = [
        "u_awac",
        "awac",
        "awac_n",
    ]

    color_map = {
            "u_awac": colors[0],
            "awac": colors[3],
            "awac_n": colors[1],
    }

        PBRL filtering plots
    color_map = {
            "rebrac": colors[5],
            "pbrl": colors[4],
            "pbrl_f": colors[1],
    }

        AWAC full eval, don't show awac-n
    algorithms = [
            "rebrac", "u_awac", "awac", "pbrl",
            ]
    color_map = {
            "rebrac": colors[5],
            "pbrl": colors[4],
            "u_awac": colors[0],
            "awac": colors[3],
    }
    """

    print("Evaluating algorithms:", algorithms)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        all_results[dataset] = {}
        
        for algorithm in algorithms:
            color = color_map[algorithm]


            df_sel = df[(df.dataset == dataset) & (df.algorithm == algorithm)]
            returns_list = df_sel["final_scores"].tolist()


            # Some entries erroneously had extra runs. Trim to first 10.
            returns_list = returns_list[:10]
            returns_array = jnp.array(returns_list)

            if len(returns_array) == 0:
                continue

            # final_scores_mean
            means_list = df_sel["final_scores_mean"].tolist()[:10]
            stds_list = df_sel["final_scores_std"].tolist()[:10]

            
            mean_of_means = np.mean(means_list)
            median_of_means = np.median(means_list)
            std_of_means = np.std(means_list)

            best_idx = np.argmax(means_list)
            median_idx = np.argsort(means_list)[len(means_list) // 2]

            best_mean = means_list[best_idx]
            best_std = stds_list[best_idx]
            median_mean = means_list[median_idx]
            median_std = stds_list[median_idx]

            # save all

            # Store info for this algorithm
            all_results[dataset][algorithm] = {
                "mean_of_means": float(mean_of_means),
                "median_of_means": float(median_of_means),
                "std_of_means": float(std_of_means),
                "best_mean": float(best_mean),
                "best_std": float(best_std),
                "median_mean": float(median_mean),
                "median_std": float(median_std),
            }

            results = bootstrap_bandit_trials(
                returns_array,
                seed=idx,
                num_subsample=5,
                num_repeats=500,
                max_pulls=100,
                ucb_alpha=2.0,
                n_bootstraps=1000,
                confidence=0.95,
            )

            ax.plot(
                results["pulls"],
                results["estimated_bests_mean"],
                label=algorithm,
                color=color,
            )
            ax.fill_between(
                results["pulls"],
                results["estimated_bests_ci_low"],
                results["estimated_bests_ci_high"],
                alpha=0.3,
                color=color
            )

        # Add grid to each subplot
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)  # Put grid behind the data
        
        ax.set_xscale("log")  # Logarithmic x-axis
        ax.set_title(f"{dataset}", fontsize=8)

    # Create legend with correct color mapping
    from matplotlib.lines import Line2D

    legend_handles = [Line2D([0], [0], color=color_map[algorithm], lw=4,
                             label=algorithm.replace('_', '-').upper(),
                             markersize=8)
                      for algorithm in algorithms]

    # Place legend BELOW the figure
    legend = fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),  # Adjust this value to position below
        ncol=len(algorithms),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        columnspacing=0.5,
        borderpad=0.75,
        edgecolor='black',
        fontsize=8,
    )

    # add x axis and y axis label
    fig.text(0.5, 0.12, 'Number of policy evaluations', ha='center', fontsize=10)
    fig.text(0.01, 0.55, 'Mean D4RL score', va='center', rotation='vertical', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Increase bottom margin for legend
    plt.savefig("figures/eval_full.pdf", dpi=300, bbox_inches='tight')

    rows = []
    for dataset, algos in all_results.items():
        for algorithm, stats in algos.items():
            row = {
                "dataset": dataset,
                "algorithm": algorithm,
                "mean_of_means": stats.get("mean_of_means", None),
                "median_of_means": stats.get("median_of_means", None),
                "std_of_means": stats.get("std_of_means", None),
                "best_mean": stats.get("best_mean", None),
                "best_std": stats.get("best_std", None),
                "median_mean": stats.get("median_mean", None),
                "median_std": stats.get("median_std", None),
            }
            rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    # Save as CSV
    df.to_csv("gt_results.csv", index=False)
