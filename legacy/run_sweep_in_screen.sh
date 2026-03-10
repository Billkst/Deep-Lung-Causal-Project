#!/usr/bin/env bash
set -e
mkdir -p logs
screen -X -S dlc_sweep quit || true
screen -dmS dlc_sweep bash -c "
source /home/UserData/miniconda/bin/activate /home/UserData/ljx/conda_envs/dlc_env
echo 'Env activated. Starting script...'
python -u src/run_parameter_sweep_final.py > logs/parameter_sweep_final_screen.log 2>&1
"
echo "Restarted with source activate. Logs at logs/parameter_sweep_final_screen.log"
