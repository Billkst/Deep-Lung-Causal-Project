#!/bin/bash
SEEDS=(42 43 44 45 46)
echo "Starting 5-seed training for Config K..."
for s in "${SEEDS[@]}"; do
   echo "Launching Seed $s..."
   conda run -p /home/UserData/ljx/conda_envs/dlc_env python src/dlc/run_final_sota.py \
      --output-tag k_seed_$s \
      --d-hidden 192 --num-layers 4 \
      --lambda-hsic 0.01 --lambda-pred 7.0 \
      --lambda-ite 0.3 --lambda-cate 0.5 \
      --lambda-prob 2.0 --lambda-adv 0.3 \
      --lambda-sens 0.05 --sens-eps-scale 0.2 \
      --epochs-pre 100 --epochs-fine 50 \
      --seed $s \
      > logs/dlc_training_k_seed_$s.log 2>&1 &
done
wait
echo "All seeds done."
