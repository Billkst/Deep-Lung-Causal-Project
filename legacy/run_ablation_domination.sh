#!/bin/bash
source activate /home/UserData/ljx/conda_envs/dlc_env
python -u src/run_rigorous_ablation.py --epochs 60
