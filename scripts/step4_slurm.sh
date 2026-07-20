#!/usr/bin/env bash
#SBATCH --job-name=step4_graph
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/step4/slurm_%j.out
#SBATCH --error=logs/step4/slurm_%j.err

set -euo pipefail

mkdir -p logs/step4
echo "SLURM job ${SLURM_JOB_ID:-unknown} started at $(date -Iseconds)"
python scripts/check_environment.py
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode full \
  --device cuda \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"
echo "SLURM job ${SLURM_JOB_ID:-unknown} finished at $(date -Iseconds)"

