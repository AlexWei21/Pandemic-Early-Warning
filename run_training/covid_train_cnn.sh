#!/bin/bash
#SBATCH -c 1
#SBATCH -t 1-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
dt=$(date +%m-%d-%H%M)
#SBATCH -o run_logs/hostname_${dt}_%j.out
#SBATCH -e run_logs/hostname_${dt}_%j.err

train_len=56

# Determine batch_size based on train_len
if [ "$train_len" -eq 56 ]; then
    batch_size=1250
elif [ "$train_len" -eq 42 ]; then
    batch_size=1285
elif [ "$train_len" -eq 28 ]; then
    batch_size=1328
elif [ "$train_len" -eq 14 ]; then
    batch_size=1353  
else
    echo "Error: Invalid train_len value. Supported values are 56, 42, 28, or 14."
    exit 1
fi

python -m run_training.run_cnn \
        --lr 1e-5 \
        --batch_size $batch_size \
        --target_training_len $train_len \
        --pred_len 84 \
        --record_run True \
        --max_epochs 500 \
        --log_dir logs/ \
        --loss Combined_Loss \
        --past_pandemics dengue ebola sars influenza \
        --use_lr_scheduler False \
        --loss_mae_weight 1 \
        --loss_mape_weight 20 \
        --output_dir output/cnn/covid_$(date +%m-%d-%H00)_$train_len-84/ \
