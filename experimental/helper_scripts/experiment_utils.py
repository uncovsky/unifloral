import os
import time
import yaml
import wandb
import warnings

def get_param(cfg, key):
    """
        Retrieves a parameter value from the given config dictionary.
        The parameter can be either a single value or a list of values.
    """
    entry = cfg["parameters"][key]
    if "value" in entry:
        return entry["value"]   
    elif "values" in entry:
        return entry["values"]            
    else:
        raise KeyError(f"No 'value' or 'values' for {key}")

def load_config(config_path):
    """
        Load a single yaml config file into memory
    """
    print(f"Loading configuration file: {config_path}")
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error loading YAML file {config_path}: {e}")
            return None


def load_configs(config_dir):
    """
        Load all yaml configs from the specified directory into memory
    """

    print("Loading configuration files...")
    paths = os.listdir(config_dir)
    paths = [os.path.join(config_dir, p) for p in paths if p.endswith(".yaml") or p.endswith(".yml")]
    print(f"Found {len(paths)} configuration files in {config_dir}")

    config_files = [load_config(p) for p in paths]
    
    
    if not config_files:
        warnings.warn("No configuration files found in the specified directory.")
        return []

    return config_files


def wandb_sweep_from_config(config, 
                            run_limit=None, 
                            environment=None,
                            project=None,
                            entity=None):
    """
        Initializes wandb.sweep from provided config
        Runs wandb agent & executes the sweep (blocking call)

        run_limit caps the number of runs this sweep will execute.

        environment optionally replaces value of dataset parameter.
        similarly, project/entity can be set to override config.
    """

    algo_name = config.get("name", "unspecified_algo")
    steps = get_param(config, "num_updates")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    assert("program" in config), "Sweep config must contain 'program' key"
    assert("parameters" in config), "Sweep config must contain 'parameters' key"
    assert("dataset-name" in config['parameters']), "Sweep config must contain 'dataset-name' parameter"

    """
        Optionally set variables in config.
    """

    if project is not None:
        config["project"] = project

    if entity is not None:
        config["entity"] = entity

    if environment is not None:
        print(f"Replacing config env with {environment}.")
        config["parameters"]["dataset-name"] = {
            "value": environment
        }

    config["name"] = f"{algo_name}-{timestamp}"
    sweep_id = wandb.sweep(config)
    print(f"Sweep created for {algo_name} with id: {sweep_id}, steps: {steps}")
    wandb.agent(sweep_id, function=None, count=run_limit)
    # blocks until agent terminates
