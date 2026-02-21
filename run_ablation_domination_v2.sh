#!/bin/bash
export PYTHONPATH=:/home/UserData/ljx/Project_1
conda run -p /home/UserData/ljx/conda_envs/dlc_env python -u src/run_rigorous_ablation.py --epochs 80
