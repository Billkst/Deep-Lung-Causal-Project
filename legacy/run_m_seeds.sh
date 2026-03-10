#!/bin/bash
SEEDS=(42 43 44 45 46)
echo "Starting 5-seed training for Config M (Pure Pred)..."
for s in "${SEEDS[@]}"; do
   echo "Launching Seed $s..."
   conda run -p /home/UserData/ljx/conda_envs/dlc_env python src/dlc/run_final_sota.py \
      --output-tag m_seed_$s \
      --d-hidden 256 --num-layers 4 \
      --lambda-hsic 0.0 --lambda-pred 1.0 \
      --lambda-ite 0.0 --lambda-cate 0.0 \
      --lambda-prob 1.0 --lambda-adv 0.0 \
      --lambda-sens 0.0 --sens-eps-scale 0.0 \
      --epochs-pre 50 --epochs-fine 50 \
      --seed $s \
      > logs/dlc_training_m_seed_$s.log 2>&1 &
done
wait
echo "All seeds done."
