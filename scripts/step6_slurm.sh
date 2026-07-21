#!/usr/bin/env bash
#SBATCH --job-name=step6_naive_news
#SBATCH --output=logs/step6/slurm_%j.out
#SBATCH --error=logs/step6/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${CONFIG:-configs/step6_naive_news.yaml}"
python -m src.news.run_step6 --config "${CONFIG}" --mode full --device cuda --resume

