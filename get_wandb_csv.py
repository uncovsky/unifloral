import argparse
import os
import pandas as pd
import wandb

# Can use filters when querying runs to get only certain env, etc.
filters = None
output_dir = "csv/"
projects = None

def save_runs(projects, filters, output_dir):

    """
        Saves all runs from projects \in projects argument into a single csv
        file output_dir/<project_name>_runs.csv

        All runs are concat into one DataFrame, with additional columns for run
        id, run name, and run index.

        filters can be provided to only consider certain runs
    """

    api = wandb.Api()
    # create output dir if it doesn't exist
    if not output_dir.endswith('/'):
        output_dir += '/'

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"wandb_runs_{timestamp}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all the data
    project_list = projects if projects is not None else wandb.Api().projects()

    for project in project_list:

        all_runs_data = []
        runs = api.runs(project.name, filters=filters)

        print("Processing project:", project.name)
        
        for i, run in enumerate(runs):
            history = run.history()

            if history.empty:
                continue

            print(f"Processing run {i+1}/{len(runs)}: {run.name} ({run.id})")

            # Add a column for run id or run index to identify runs
            history["run_id"] = run.id
            history["run_name"] = run.name
            history["run_index"] = i
            all_runs_data.append(history)
        
        if all_runs_data:
            # Concatenate all runs data into one DataFrame
            project_df = pd.concat(all_runs_data, ignore_index=True)
            # only keep the project name & save
            project_name = project.name.split("/")[-1]
            csv_file_path = os.path.join(output_dir, f"{project_name}_runs.csv")
            project_df.to_csv(csv_file_path, index=False)

    print(f"Runs data saved to {output_dir}")

if __name__ == "__main__":
    # add argument for env name
    parser = argparse.ArgumentParser(description="Fetch and save WandB runs data to CSV.")
    parser.add_argument("--projects", nargs="+", default=projects, help="List of WandB project paths to fetch runs from.")
    parser.add_argument("--filters", type=str, default=None, help="Filters to apply when querying runs (e.g., 'env=CartPole-v1').")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Directory to save the output CSV files.")

    args = parser.parse_args()

    # Parse filters if provided
    if args.filters:
        filters = eval(args.filters)  # Convert string to dictionary if needed
    else:
        filters = None


    # Call the function to save runs
    save_runs(args.projects, filters, args.output_dir)

    print("Done!")


    
