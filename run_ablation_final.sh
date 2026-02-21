#!/bin/bash
set -e
conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env python -u src/run_rigorous_ablation.py --epochs 50 --seeds 42 43 44 45 46 --sota-lambda-cate 10.0 --sota-lambda-pred 6.5 --sota-lambda-hsic 0.005 --sota-lambda-ite 15.0
