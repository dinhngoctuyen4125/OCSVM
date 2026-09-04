#!/bin/bash

#SBATCH --job-name=prod
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

export CUDA_VISIBLE_DEVICES=0

/home/ritsu/miniconda3/envs/prod_eval/bin/python train_ood.py \
    --model_name_or_path "tummitum/codebert-deprecated" \
    --data_path "../Data-Collection/codellama/D_forget.json" \
    --output_dir "./ckpt" \
    --seed 2026
