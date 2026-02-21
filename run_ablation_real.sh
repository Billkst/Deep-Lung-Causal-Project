#!/bin/bash
# Direct python execution to avoid conda run IO buffering issues
/home/UserData/ljx/conda_envs/dlc_env/bin/python -u src/run_rigorous_ablation.py --epochs 100 --seeds 42 43 44 45 46
