#!/bin/bash
set -e

mkdir -p ~/.mujoco/

cd ~/.mujoco/

wget --content-disposition https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

# Extract and clean up
tar --no-same-owner -xvzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz

# Set environment variables for MuJoCo (add to .bashrc if you want it permanent)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210

# Install mjrl from GitHub
if [ ! -d "$HOME/mjrl" ]; then
    git clone https://github.com/aravindr93/mjrl.git $HOME/mjrl
else
    echo "mjrl repo already cloned at $HOME/mjrl"
fi

pip install -e $HOME/mjrl
