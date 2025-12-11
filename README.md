# Uncertainty-based Offline Reinforcement Learning

This repository implements a generic algorithmic framework for ensemble-based offline RL, which enables easy experimentation and ablation of key design choices of different algorithms.
This framework is the subject of my Master's thesis, available [here](https://is.muni.cz/auth/th/dxwqj/).


The framework generalizes several offline RL algorithms, namely:

- **PBRL** — [Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning](https://arxiv.org/abs/2202.11566)
- **MSG** — [Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters](https://arxiv.org/abs/2205.13703)
- **SAC-N, EDAC** — [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548) 

And can be extended to accommodate other modifications of these algorithms with other regularization terms, such as
- **RORL** — [RORL: Robust Offline Reinforcement Learning via Conservative Smoothing](https://arxiv.org/abs/2206.02829)


The framework and evaluation protocol are built upon the Unifloral library available [here](https://github.com/EmptyJackson/unifloral). 

To see how the algorithms are implemented inside the framework, refer to the last section of this README.


## Setting up the container:

To set up and run the framework using Docker, use the provided Makefile commands:

```bash
make build   # Builds the Docker image
```
And then either:
```bash
make up   # Launches the container without GPU access
```
or:
```bash
make up-gpu   # Launches w. gpu device 0, requires nvidia container toolkit
```
## Running the experiments 

You can rerun all of the experiments from the thesis (sequentially) like so:
```bash
python experiments/run_experiments --entity YOUR_WANDB_ENTITY --project YOUR_WANDB_PROJECT
```
Note that the experiments are set up via wandb sweeps, which requires you to have a wandb account. A single run of a selected algorithm can be executed as follows:
``` bash
python algorithms/unified.py --critic_regularizer [msg|pbrl|cql] --critic_lagrangian lambda --dataset_name [hopper-medium-v2, ...]
```
You can also use the help flag for parameter names and their semantics. 
``` bash
python algorithms/unified.py --help
```

## Structure of the Repository
All of the code is located in the directory `ensemble_offline_rl`

- The subdirectory `algorithms` contains the unified algorithmic framework `unified.py` + modified BC/ReBRAC baselines from Unifloral used for evaluation.
- The subdirectory `configs` contains yaml configuration files to instantiate all of the baselines inside our unified framework.
- The subdirectory `experiments` contains all of the wandb sweep configs with hyperparameter spaces used to obtain the results in the thesis.
- The subdirectory `infra` contains the infrastructure for the framework, which contains:
    - `checkpoints/` utils for checkpointing models
    - `dataset/` generic dataset wrapper, which allows us to handle both minari and D4RL environments
    - `ensemble_training/` implementations of critic regularization losses (MSG/EDAC/CQL, ..), can add new regularization terms, etc. here
    - `models/` - actor and critic networks, taken from unifloral + added randomized prior network support
    - `utils/` - logging for visualizations, diversity of ensemble predictions, scheduling of Lagrangians, etc.
- The subdirectory `results` contains all the data from experiments + visualization scripts, and figures from the thesis.

### Summary of Training Loop
The training logic shared by all algorithms is implemented in the function *make_train_step()*  `algorithms/unified.py`. This training loop is instantiated with selected critic and ensemble (diversity) regularization terms, which 
are specified via CLI arguments and implemented in their respective `infra/` entries. To see how different hyperparameters from the original algorithms map to parameters of the framework, refer to the `configs` directory.



