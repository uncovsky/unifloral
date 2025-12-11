import pandas as pd


# Load your CSV
df = pd.read_csv("gt_results.csv")

"""
    modify the lines below to restrict algorithms accordingly
"""

# Full eval
algos = ["bc","cql","edac","msg","pbrl","rebrac","sac_n"]

# AWAC eval
#algos = ["pbrl","rebrac","awac", "u_awac"]

latex = []
latex.append(r"\begin{table}[ht]")
latex.append(r"\centering")
latex.append(r"\caption{Best mean performance ($\pm$ std) per dataset and algorithm.}")
latex.append(r"\sisetup{separate-uncertainty = true, table-align-uncertainty = true}")
latex.append(r"\begin{tabular}{l" + "S[table-format=3.2(2)]"*len(algos) + "}")
latex.append(r"\toprule")

# Header
header = ["Dataset"] + [algo.upper() for algo in algos]
latex.append(" & ".join(header) + r" \\")
latex.append(r"\midrule")

datasets = df["dataset"].unique()

for dataset in datasets:
    row = [dataset]
    df_ds = df[df["dataset"] == dataset]
    print(df_ds)
    
    # Determine the best algorithm by best_mean
    best_idx = df_ds["best_mean"].idxmax()
    
    for algo in algos:
        df_algo = df_ds[df_ds["algorithm"] == algo]
        if not df_algo.empty:
            best = df_algo["best_mean"].values[0]
            best_std = df_algo["best_std"].values[0]
            cell = f"{best:.2f} \pm {best_std:.2f}"
            row.append(cell)
        else:
            row.append("{}")
    
    latex.append(" & ".join(row) + r" \\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

# Save LaTeX table
with open("figures/best_mean_table.tex", "w") as f:
    f.write("\n".join(latex))

# Median policy table

latex = []
latex.append(r"\begin{table}[ht]")
latex.append(r"\centering")
latex.append(r"\caption{Best mean performance ($\pm$ std) per dataset and algorithm.}")
latex.append(r"\sisetup{separate-uncertainty = true, table-align-uncertainty = true}")
latex.append(r"\begin{tabular}{l" + "S[table-format=3.2(2)]"*len(algos) + "}")
latex.append(r"\toprule")

# Header
header = ["Dataset"] + [algo.upper() for algo in algos]
latex.append(" & ".join(header) + r" \\")
latex.append(r"\midrule")

datasets = df["dataset"].unique()
for dataset in datasets:
    row = [dataset]
    df_ds = df[df["dataset"] == dataset]
    
    for algo in algos:
        df_algo = df_ds[df_ds["algorithm"] == algo]
        if not df_algo.empty:
            best = df_algo["median_mean"].values[0]
            best_std = df_algo["median_std"].values[0]
            cell = f"{best:.2f} \pm {best_std:.2f}"
            row.append(cell)
        else:
            row.append("{}")
    
    latex.append(" & ".join(row) + r" \\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

# Save LaTeX table
with open("figures/median_mean_table.tex", "w") as f:
    f.write("\n".join(latex))


# Median policy table

latex = []
latex.append(r"\begin{table}[ht]")
latex.append(r"\centering")
latex.append(r"\caption{Best mean performance ($\pm$ std) per dataset and algorithm.}")
latex.append(r"\sisetup{separate-uncertainty = true, table-align-uncertainty = true}")
latex.append(r"\begin{tabular}{l" + "S[table-format=3.2(2)]"*len(algos) + "}")
latex.append(r"\toprule")

# Header
header = ["Dataset"] + [algo.upper() for algo in algos]
latex.append(" & ".join(header) + r" \\")
latex.append(r"\midrule")

datasets = df["dataset"].unique()
for dataset in datasets:
    row = [dataset]
    df_ds = df[df["dataset"] == dataset]
    
    for algo in algos:
        df_algo = df_ds[df_ds["algorithm"] == algo]
        if not df_algo.empty:
            best = df_algo["mean_of_means"].values[0]
            best_std = df_algo["std_of_means"].values[0]
            cell = f"{best:.2f} \pm {best_std:.2f}"
            row.append(cell)
        else:
            row.append("{}")
    
    latex.append(" & ".join(row) + r" \\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

# Save LaTeX table
with open("figures/aggregate_table.tex", "w") as f:
    f.write("\n".join(latex))
