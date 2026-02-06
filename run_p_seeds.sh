#!/bin/bash
# Run Config P (Optimized for CATE & AUC, Low Params) on 5 seeds

set -e

echo "Starting Config P (Optimized) Run on 5 seeds..."

PYTHON="python"

for SEED in 42 43 44 45 46; do
    echo "----------------------------------------------------------------"
    echo "Running Config P with SEED=$SEED"
    echo "----------------------------------------------------------------"
    
    # Using run_final_sota_p.py, logging to logs/
    $PYTHON src/dlc/run_final_sota_p.py \
        --output-tag "p_seed_${SEED}" \
        --seed $SEED \
        --d-hidden 128 \
        --num-layers 4 \
        --lambda-hsic 0.1 \
        --lambda-pred 2.0 \
        --lambda-ite 1.0 \
        --lambda-cate 2.0 \
        --lambda-prob 1.0 \
        --epochs-pre 130 \
        --epochs-fine 60 \
        > logs/run_p_seed_${SEED}.log 2>&1
done

echo "Done. Results in results/final_sota_metrics_p_seed_*.json"
