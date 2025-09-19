import argparse
import os
import pandas as pd
import wandb

# Can use filters when querying runs to get only certain env, etc.
filters = None
output_dir = "csv/"
projects = None


def create_directory(output_dir):
    if not output_dir.endswith('/'):
        output_dir += '/'

    # do not timestamp lol
    #timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    #output_dir = os.path.join(output_dir, f"wandb_runs_{timestamp}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    return output_dir

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
    output_dir_timestamped = create_directory(output_dir)

    # Get all the data
    if projects is None:
        # If no projects are specified, fetch all projects
        project_list = api.projects()

    elif isinstance(projects, str):
        project_list = [p for p in api.projects() if p.name == projects]

    elif isinstance(projects, list):
        print(projects)
        project_list = [p for p in api.projects() if p.name in projects]
    else:
        raise ValueError("projects must be a string or a list of strings.")

    print("Found projects:")
    print(project_list)

    for project in project_list:

        log_dir_path = os.path.join(output_dir_timestamped, f"{project.name}/logs")
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        csv_path = os.path.join(output_dir_timestamped, f"{project.name}/data")
        print(csv_path)
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)

        all_runs_data = []
        runs = api.runs(project.name, filters=filters)

        first_config = None

        print("Processing project:", project.name)
        print("Logs saved to", log_dir_path)
        print("Got {} runs".format(len(runs)))
        for i, run in enumerate(runs):
            history = run.history()

            config = run.config
            logs = run.summary

            if history.empty:
                print(f"Skipping run {i+1}/{len(runs)}: {run.name} ({run.id}), no data")
                print(f"{50* '-'}")
                continue

            print(f"Processing run {i+1}/{len(runs)}: {run.name} ({run.id})")

            if first_config is None:
                first_config = config

            elif first_config != config:

                # find values that are different
                diff = {k: v for k, v in config.items() if k not in first_config or first_config[k] != v}
                last_diff = {k: v for k, v in first_config.items() if k in diff}
                
                print(f"    Config changed in run {run.id}: {diff} vs {last_diff}")

            # Add a column for run id or run index to identify runs
            history["run_id"] = run.id
            history["run_name"] = run.name
            history["run_index"] = i

            all_runs_data.append(history)

            logs_file_path = os.path.join(log_dir_path, f"run_{i}_{run.id}_logs.txt")
            # Save logs to a text file
            with open(logs_file_path, 'w') as f:
                for key, value in logs.items():
                    f.write(f"{key}: {value}\n")

            print(f"{50* '-'}")
        
        project_name = project.name.split("/")[-1]
        csv_file_path = os.path.join(csv_path, f"{project_name}_runs.csv")

        if all_runs_data:
            # Concatenate all runs data into one DataFrame
            project_df = pd.concat(all_runs_data, ignore_index=True)
            # only keep the project name & save
            project_df.to_csv(csv_file_path, index=False)


        # save config as text file
        config_file_path = os.path.join(output_dir_timestamped, f"{project_name}__config.txt")
        with open(config_file_path, 'w') as f:
            f.write(str(first_config))

        print(f"Project {project.name} runs saved to {csv_file_path}")

    print("Runs saved to CSV files in directory:", output_dir_timestamped)


if __name__ == "__main__":
    # add argument for env name
    parser = argparse.ArgumentParser(description="Fetch and save WandB runs data to CSV.")
    parser.add_argument("--projects", nargs="+", default=projects, help="List of WandB project paths to fetch runs from.")
    parser.add_argument("--filters", type=str, default=None, help="Filters to apply when fetching runs from projects")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Directory to save the output CSV files.")


    """ example filter 
            {"config.env_name": {"$regex": "^antmaze_"}
    """

    args = parser.parse_args()

    # Parse filters if provided
    if args.filters:
        filters = eval(args.filters)  # Convert string to dictionary if needed
    else:
        filters = None


    # Call the function to save runs
    save_runs(args.projects, filters, args.output_dir)

    print("Done!")


    
