import pandas as pd

# Load csv outputted by evaluation.py script
# and generate tex table with median/best perf
df = pd.read_csv("full_results.csv")

"""
    modify the lines below to restrict algorithms accordingly
"""

# Full eval
algos = ["bc","cql","edac","msg","pbrl","rebrac","sac_n"]

# AWAC eval
#algos = ["pbrl","rebrac","awac", "u_awac"]

def generate_table(df, algos, value_col, std_col, filename, caption):
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{{caption}}}")
    latex.append(r"\sisetup{separate-uncertainty = true, table-align-uncertainty = true}")
    latex.append(r"\begin{tabular}{l" + "S[table-format=3.2(2)]"*len(algos) + "}")
    latex.append(r"\toprule")

    # Header
    header = ["Dataset"] + [algo.upper() for algo in algos]
    latex.append(" & ".join(header) + r" \\")
    latex.append(r"\midrule")

    datasets = df["dataset"].unique()
    algo_means = {algo: [] for algo in algos}  # to collect values for overall mean

    for dataset in datasets:
        row = [dataset]
        df_ds = df[df["dataset"] == dataset]

        for algo in algos:
            df_algo = df_ds[df_ds["algorithm"] == algo]
            if not df_algo.empty:
                val = df_algo[value_col].values[0]
                std = df_algo[std_col].values[0]
                cell = f"{val:.2f} \pm {std:.2f}"
                row.append(cell)
                algo_means[algo].append(val)
            else:
                row.append("{}")
        
        latex.append(" & ".join(row) + r" \\")

    # Add Overall row
    overall_row = ["Overall"]
    for algo in algos:
        if algo_means[algo]:
            overall_mean = sum(algo_means[algo]) / len(algo_means[algo])
            overall_row.append(f"{overall_mean:.2f}")
        else:
            overall_row.append("{}")

    latex.append(r"\midrule")
    latex.append(" & ".join(overall_row) + r" \\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    # Save LaTeX table
    with open(f"figures/{filename}", "w") as f:
        f.write("\n".join(latex))

# Generate best_mean table
generate_table(
    df, algos,
    value_col="best_mean",
    std_col="best_std",
    filename="best_mean_table.tex",
    caption="Best mean performance ($\\pm$ std) per dataset and algorithm."
)

# Generate median_mean table
generate_table(
    df, algos,
    value_col="median_mean",
    std_col="median_std",
    filename="median_mean_table.tex",
    caption="Median mean performance ($\\pm$ std) per dataset and algorithm."
)

# Generate mean_of_means table
generate_table(
    df, algos,
    value_col="mean_of_means",
    std_col="std_of_means",
    filename="aggregate_table.tex",
    caption="Mean of means performance ($\\pm$ std) per dataset and algorithm."
)
