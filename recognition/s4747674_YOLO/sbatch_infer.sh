#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --job-name=isic-yolo-infer
#SBATCH --time=30:00
#SBATCH --output=recognition/s4747674_YOLO/logs/infer_%j.out

set -euo pipefail

mkdir -p recognition/s4747674_YOLO/logs

conda activate isic-yolo

cd "$SLURM_SUBMIT_DIR"

python -m recognition.s4747674_YOLO.predict \
  --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt \
  --source recognition/s4747674_YOLO/prepared/images/test \
  --save \
  --project recognition/s4747674_YOLO/inference_runs

