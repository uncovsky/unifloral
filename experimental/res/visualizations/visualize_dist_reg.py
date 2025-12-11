import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tex_setup import set_size

"""
    Plots comparing action distribution shift
    across different regularizers and lagrangian values.
"""


df = pd.read_csv("lineplots/antmaze-collapse_runs.csv")


reg_df = df[df.run_index < 6]

print(reg_df.columns)

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
        palette="deep",
        ax=ax
    )

    if reg == "cql":
        reg = "msg"

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
    palette="deep",
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
plt.savefig("figures/antmaze_dist_reg_comparison.pdf",
            bbox_inches='tight', dpi=300)



"""
    PBRL / MSG on pen-human
"""

df_pbrl = pd.read_csv("lineplots/pbrl_pen_human_dist.csv")
df_msg  = pd.read_csv("lineplots/msg_pen_human_dist.csv")


def extract_lagrangians(df, lag_values):
    long_df = []
    for lag in lag_values:
        # Match column formatting: 1 → "1" and 0.1 → "0.1"
        col_lag = int(lag) if float(lag).is_integer() else lag
        col = f"critic_lagrangian: {col_lag} - mean_action_dist"

        if col not in df.columns:
            print(f"Warning: {col} not found.")
            continue

        temp = pd.DataFrame({
            "_step": df["Step"],
            "critic_lagrangian": float(lag),
            "mean_action_dist": df[col]
        })
        long_df.append(temp)

    return pd.concat(long_df, ignore_index=True)


# λ-values
pbrl_lags = [0.01, 0.2]
msg_lags  = [0.1, 0.5, 1.0]

df_pbrl_long = extract_lagrangians(df_pbrl, pbrl_lags)
df_msg_long  = extract_lagrangians(df_msg,  msg_lags)

# Filter range (10–30)
df_pbrl_long = df_pbrl_long[(df_pbrl_long._step >= 10) & (df_pbrl_long._step <= 30)]
df_msg_long  = df_msg_long[(df_msg_long._step  >= 10) & (df_msg_long._step  <= 30)]

# Smooth
df_pbrl_long["mean_action_dist"] = df_pbrl_long["mean_action_dist"].rolling(10, min_periods=1).mean()
df_msg_long["mean_action_dist"]  = df_msg_long["mean_action_dist"].rolling(10, min_periods=1).mean()

# Shared y-limit
ylim = max(df_pbrl_long.mean_action_dist.max(),
           df_msg_long.mean_action_dist.max()) + 0.1

# Plot
fig_dims = set_size(width_fraction=0.7, height_fraction=0.2)
fig, axes = plt.subplots(1, 2, figsize=fig_dims, sharey=True, sharex=True)


# ---- PBRL subplot ----
sns.lineplot(
    data=df_pbrl_long,
    x="_step",
    y="mean_action_dist",
    hue="critic_lagrangian",
    palette="deep",
    ax=axes[0]
)
axes[0].set_title("PBRL", fontsize=8)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel(r"$\| a' - a_t \|^2_2$")
axes[0].set_ylim(0, ylim)

pbrl_handles, pbrl_labels = axes[0].get_legend_handles_labels()
axes[0].legend_.remove()


# ---- MSG subplot ----
sns.lineplot(
    data=df_msg_long,
    x="_step",
    y="mean_action_dist",
    hue="critic_lagrangian",
    palette="deep",
    ax=axes[1]
)
axes[1].set_title("MSG", fontsize=8)
axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0, ylim)

msg_handles, msg_labels = axes[1].get_legend_handles_labels()
axes[1].legend_.remove()


# ---- PBRL Legend ----
pbrl_legend = fig.legend(
    pbrl_handles, pbrl_labels,
    title=r"$\lambda$",
    loc="upper center",
    bbox_to_anchor=(0.33, 1.05),
    ncol=len(pbrl_labels),
    frameon=False,
    handlelength=1.2,
    handletextpad=0.4,
    columnspacing=0.8
)
pbrl_legend.get_title().set_fontsize(8)
for t in pbrl_legend.get_texts():
    t.set_fontsize(7)


# ---- MSG Legend ----
msg_legend = fig.legend(
    msg_handles, msg_labels,
    title=r"$\lambda$",
    loc="upper center",
    bbox_to_anchor=(0.77, 1.05),
    ncol=len(msg_labels),
    frameon=False,
    handlelength=1.2,
    handletextpad=0.4,
    columnspacing=0.8
)
msg_legend.get_title().set_fontsize(8)
for t in msg_legend.get_texts():
    t.set_fontsize(7)


# Layout
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig("figures/pbrl_msg_human_dist.pdf",
            bbox_inches="tight", dpi=300)
