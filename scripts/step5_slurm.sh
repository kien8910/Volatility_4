#!/usr/bin/env bash
#SBATCH --job-name=step5_regime_graph
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/step5/slurm_%j.out
#SBATCH --error=logs/step5/slurm_%j.err

set -euo pipefail

mkdir -p logs/step5
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode full \
  --device cuda \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --resume

