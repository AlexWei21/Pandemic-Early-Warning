#!/bin/bash
#SBATCH -c 1
#SBATCH -t 2-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
dt=$(date +%m-%d-%H%M)
#SBATCH -o run_logs/hostname_${dt}_%j.out
#SBATCH -e run_logs/hostname_${dt}_%j.err

train_len=14

# Determine batch_size based on train_len
if [ "$train_len" -eq 56 ]; then
    batch_size=260
elif [ "$train_len" -eq 42 ]; then
    batch_size=260
elif [ "$train_len" -eq 28 ]; then
    batch_size=260
elif [ "$train_len" -eq 14 ]; then
    batch_size=260  
else
    echo "Error: Invalid train_len value. Supported values are 56, 42, 28, or 14."
    exit 1
fi

python -m run_training.train_self-tune \
        --lr 1e-5 \
        --batch_size $batch_size \
        --target_training_len $train_len \
        --pred_len 84 \
        --record_run True \
        --max_epochs 1000 \
        --log_dir logs/ \
        --loss Combined_Loss \
        --dropout 0.0 \
        --include_death False \
        --population_weighting False \
        --use_scheduler False \
        --loss_mae_weight 1 \
        --loss_mape_weight 20 \
        --output_dir output/self_tune/covid_$(date +%m-%d-%H00)_$train_len-84/ \