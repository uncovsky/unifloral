import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tex_setup import set_size

df = pd.read_csv("sparse_plots/antmaze-collapse_runs.csv")




reg_df = df[df.run_index < 6]


cols = ["critic_regularizer", "_step", "critic_lagrangian", "reg_lagrangian", "mean_action_dist"]

reg_df = reg_df[cols][(reg_df._step <= 30) & (reg_df._step >= 10)]
ylim = reg_df["mean_action_dist"].max() + 0.1
# plot mean_action_dist vs _step for different critic_regularizer values
fig_dims = set_size(width_fraction=1.0, height_fraction=0.2)
print(fig_dims)

fig, axes = plt.subplots(1, 3, figsize=fig_dims, sharey=True, sharex=True)
axes = axes.flatten()

# Plot first two regularizers (CQL and PBRL)
for idx, reg in enumerate(["cql", "pbrl"]):
    ax = axes[idx]
    plot_df = reg_df[reg_df.critic_regularizer == reg].copy()

    # Smooth
    plot_df["mean_action_dist"] = \
        plot_df["mean_action_dist"].rolling(window=10, min_periods=1).mean()

    hue = "critic_lagrangian"

    sns.lineplot(
        data=plot_df,
        x="_step",
        y="mean_action_dist",
        hue=hue,
        ax=ax
    )

    ax.set_title(f"{reg.upper()}", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$\| a' - a_t \|^2_2$")
    ax.set_ylim(0, ylim)
    ax.legend_.remove()

# Plot EDAC in the third subplot
edac_df = df[(df.run_index >= 6) & (df.run_index <= 7)]
edac_df = edac_df[cols][(edac_df._step <= 30) & (edac_df._step >= 10)]

edac_df["mean_action_dist"] = \
    edac_df["mean_action_dist"].rolling(window=10, min_periods=1).mean()

sns.lineplot(
    data=edac_df,
    x="_step",
    y="mean_action_dist",
    hue="reg_lagrangian",
    ax=axes[2]
)

axes[2].set_title("EDAC", fontsize=8)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("$\| a' - a_t \|^2_2$")
axes[2].set_ylim(0, ylim)

# Get EDAC legend handles and labels before removing the legend
edac_handles, edac_labels = axes[2].get_legend_handles_labels()
axes[2].legend_.remove()

# Create shared legend for first two plots (centered above first two subplots)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="$\lambda$", loc="upper center",
           ncol=len(labels), bbox_to_anchor=(0.4, 1.05), frameon=False)

# Create separate legend for EDAC above its subplot
fig.legend(edac_handles, edac_labels, title="$\lambda$", loc="upper center",
           ncol=len(edac_labels), bbox_to_anchor=(0.84, 1.05), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space at top for legends
plt.savefig("sparse_plots/antmaze_dist_reg_comparison.pdf",
            bbox_inches='tight', dpi=300)
