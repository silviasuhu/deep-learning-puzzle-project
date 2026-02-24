#!/bin/bash
#SBATCH -J run_inference
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -p emb-gpu

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data/Software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/Software/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/data/Software/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/data/Software/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate puzzle-project

python run_inference.py