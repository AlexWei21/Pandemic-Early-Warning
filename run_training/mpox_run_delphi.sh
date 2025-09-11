#!/bin/bash
#SBATCH -c 20
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=32G
dt=$(date +%m-%d-%H%M)
#SBATCH -o run_logs/hostname_${dt}_%j.out
#SBATCH -e run_logs/hostname_${dt}_%j.err

python -m run_training.mpox_delphi_with_case