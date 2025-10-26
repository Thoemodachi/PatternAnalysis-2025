#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --job-name=isic-yolo-val
#SBATCH --time=5:00:00
#SBATCH --output=recognition/s4747674_YOLO/logs/val_%j.out

set -euo pipefail

mkdir -p recognition/s4747674_YOLO/logs

conda activate isic-yolo

cd "$SLURM_SUBMIT_DIR"

python -m recognition.s4747674_YOLO.train \
  --data-root /home/groups/comp3710/ISIC2018 \
  --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt \
  --val-only

