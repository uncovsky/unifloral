import os
from datetime import datetime
from flax.training import checkpoints


def get_experiment_dirname(args):
    """
        Maps args to a directory name which will be used to store 
        checkpoints and final returns.

        we will save the data in 
            checkpoint_dir/{experiment_dir}/checkpoints
            ,and 
            checkpoint_dir/{experiment_dir}/final_returns respectively.

    """
    name = f"{args.algorithm}_{args.dataset_name}"
    filtered_name = name.replace("/", "_").replace(".", "_")
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filtered_name = f"{filtered_name}/{time}"
    filtered_name = f"{args.checkpoint_dir}/{filtered_name}"

    return filtered_name

def create_checkpoint_dir(exp_dir):
    ckpt_dir = exp_dir + "/checkpoints"
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir

def save_train_state(train_state, ckpt_dir, step):
    checkpoints.save_checkpoint(ckpt_dir, target=train_state, step=step,
                                overwrite=False, keep=2)
    print(f"Checkpoint saved at step {step} in {ckpt_dir}")
