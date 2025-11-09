import matplotlib.pyplot as plt
from tex_setup import set_size
import pandas as pd
import seaborn as sns


# given a df and and an x and y column, plot a lineplot with tex setup

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
        df = pd.read_csv(f"lineplots/edac{x}.csv")
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

    fig.savefig("lineplots/edac_lineplot.pdf", bbox_inches='tight', dpi=300)


