#!/bin/bash
SEEDS=(42 43 44 45 46)
echo "Starting 5-seed training for Config N (Balanced SOTA)..."
for s in "${SEEDS[@]}"; do
   echo "Launching Seed $s..."
   conda run -p /home/UserData/ljx/conda_envs/dlc_env python src/dlc/run_final_sota.py \
      --output-tag n_seed_$s \
      --d-hidden 128 --num-layers 3 \
      --lambda-hsic 0.05 \
      --lambda-pred 4.0 \
      --lambda-ite 1.0 \
      --lambda-cate 0.5 \
      --lambda-prob 1.2 \
      --lambda-adv 0.5 \
      --lambda-sens 0.1 \
      --sens-eps-scale 0.1 \
      --epochs-pre 100 --epochs-fine 50 \
      --seed $s \
      > logs/dlc_training_n_seed_$s.log 2>&1 &
done
wait
echo "All seeds done."
