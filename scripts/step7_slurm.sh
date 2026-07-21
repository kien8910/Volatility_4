#!/usr/bin/env bash
#SBATCH --job-name=step7_pilot
#SBATCH --output=logs/step7/slurm_%j.out
#SBATCH --error=logs/step7/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

CONFIG="${CONFIG:-configs/step7_stock_specific_news.yaml}"
python -m src.stock_news_impact.run_step7 --config "${CONFIG}" --mode pilot --device cuda --resume

