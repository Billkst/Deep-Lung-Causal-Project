#!/usr/bin/env bash
set -e
mkdir -p logs
# Kill existing session if any
screen -X -S dlc_sweep quit || true

echo "Starting Parameter Sweep in screen session 'dlc_sweep'..."
screen -dmS dlc_sweep bash -c "
/home/UserData/ljx/conda_envs/dlc_env/bin/python -u src/run_parameter_sweep_final.py > logs/parameter_sweep_final_screen.log 2>&1
"
echo "Session started. View with: screen -r dlc_sweep"
echo "Logs being written to: logs/parameter_sweep_final_screen.log"
