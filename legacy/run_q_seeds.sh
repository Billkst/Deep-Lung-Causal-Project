#!/bin/bash
# Run Config Q (High AUC + Good CATE) on 5 seeds

set -e

echo "Starting Config Q Run on 5 seeds..."

PYTHON="python"

for SEED in 42 43 44 45 46; do
    echo "----------------------------------------------------------------"
    echo "Running Config S with SEED=$SEED"
    echo "----------------------------------------------------------------"
    
    $PYTHON src/dlc/run_final_sota_q.py \
        --output-tag "s_seed_${SEED}" \
        --seed $SEED \
        --d-hidden 128 \
        --num-layers 3 \
        --lambda-hsic 0.1 \
        --lambda-pred 3.5 \
        --lambda-ite 1.0 \
        --lambda-cate 2.0 \
        --epochs-pre 200 \
        --epochs-fine 100 \
        > logs/run_s_seed_${SEED}.log 2>&1
done

echo "Done. Results in results/final_sota_metrics_s_seed_*.json"
