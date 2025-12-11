import matplotlib.pyplot as plt
from tex_setup import set_size
import pandas as pd
import seaborn as sns

"""
    Generic line plotting for figures
"""


def plot_lineplot(df, x_col, y_col, hue_col=None, title=None, xlabel=None,
                  ylabel=None, figsize=None):


    if figsize is None:
        figsize = set_size(width_fraction=1.0, height_fraction=0.2)
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette="deep",
        ax=ax
    )
    if title is not None:
        ax.set_title(title, fontsize=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # remove old legend
    ax.legend_.remove()
    # fancy legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title=f"${hue_col}$", loc="upper center",
               ncol=len(labels), bbox_to_anchor=(0.5, 1.15), frameon=False)

    return fig, ax


if __name__ == "__main__":

    """
        EDAC lineplots
    """
    # load to big df
    df_list = []
    for x in [10, 20, 30]:
        df = pd.read_csv(f"../vis_data/lineplots/edac{x}.csv")
        df["N"] = x
        df["Step"] = df["Step"] * (14 / 4)

        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    print(df)


    fig, ax = plot_lineplot(
        df,
        x_col="Step",
        y_col="dataset-name: bandit_30-v0 - ensemble_regularizer_loss",
        hue_col="N",
        xlabel="Epochs",
        ylabel = r"$\mathcal{R}_{\mathrm{EDAC}}$",
        figsize=set_size(width_fraction=0.4, height_fraction=0.15)
    )

    fig.savefig("figures/edac_lineplot.pdf", bbox_inches='tight', dpi=300)


    """
        UWAC and AWAC Q-plots
    """
    df_awac = pd.read_csv("../vis_data/awac_plots/awac_q_values_pen_human.csv")
    df_uwac = pd.read_csv("../vis_data/awac_plots/uwac_q_values_pen_human.csv")

    # Select all sweeps
    awac_mean_cols = [col for col in df_awac.columns if "q_pred_mean" in col and "__" not in col]
    uwac_mean_cols = [col for col in df_uwac.columns if "q_pred_mean" in col and "__" not in col]
    df_awac['q_mean_avg'] = df_awac[awac_mean_cols].mean(axis=1)
    df_uwac['q_mean_avg'] = df_uwac[uwac_mean_cols].mean(axis=1)
    df_merged = pd.merge(df_awac[['Step', 'q_mean_avg']], 
                         df_uwac[['Step', 'q_mean_avg']], 
                         on='Step', 
                         suffixes=('_awac', '_uwac'))

    df_awac['Method'] = 'AWAC'
    df_uwac['Method'] = 'U-AWAC'
    df_both = pd.concat([df_awac[['Step','q_mean_avg','Method']],
                         df_uwac[['Step','q_mean_avg','Method']]], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    sns.lineplot(ax=axes[0], data=df_both[df_both['Method']=='AWAC'],
                 x='Step', y='q_mean_avg', label='AWAC')
    axes[0].set_title('AWAC Q-values')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Q-value')

    sns.lineplot(ax=axes[1], data=df_both[df_both['Method']=='U-AWAC'],
                 x='Step', y='q_mean_avg', label='U-AWAC', color='orange')
    axes[1].set_title('U-AWAC Q-values')
    axes[1].set_xlabel('Step')

    plt.tight_layout()
    plt.show()

