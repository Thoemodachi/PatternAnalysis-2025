# ISIC Lesion Detection Workflow
This package trains and evaluates a YOLOv8 detector on the ISIC 2018 dermoscopy dataset. It converts the segmentation masks into YOLO-ready bounding boxes, launches Ultralytics training, and runs inference for qualitative checks. The current workflow operates as a single-class lesion detector driven solely by the segmentation masks.

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas matplotlib scikit-image pyyaml
```

## Dataset Preparation & Training
```bash
python -m recognition.s4747674_YOLO.train \
  --data-root /home/groups/comp3710/ISIC2018 \
  --model yolov8m.pt \
  --epochs 100 \
  --batch 32 \
  --imgsz 640 \
  --project recognition/s4747674_YOLO/runs \
  --name yolov8m_isic
```
This command will
- build `prepared/` inside `--work-dir` (default: `recognition/s4747674_YOLO/prepared`) with YOLO-formatted images, labels, and `isic.yaml`;
- train the YOLO model (resumable via `--resume`);
- generate `results.csv` plus loss/metric plots in the Ultralytics run directory.

## Validation Only
```bash
python -m recognition.s4747674_YOLO.train \
  --data-root /home/groups/comp3710/ISIC2018 \
  --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt \
  --val-only
```

## Inference & Visualisation
```bash
python -m recognition.s4747674_YOLO.predict \
  --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt \
  --source recognition/s4747674_YOLO/prepared/images/test \
  --save \
  --project recognition/s4747674_YOLO/inference_runs
```
Predictions are written to the directory supplied via `--project`/`--name` (e.g., `recognition/s4747674_YOLO/inference_runs/yolov8m_isic`).

## Helpful Flags
- `--force`: rebuild the YOLO dataset even if it already exists.
- `--device cuda:0`: choose GPU or CPU explicitly.
- `--conf 0.4`: adjust confidence threshold when running inference.
- `--export`: available via `YOLODetector.export` if you need ONNX/TensorRT exports.
- The default dataset builder assumes a single `lesion` class. Extend `ISICDatasetConfig.class_names` and the loader if you later incorporate per-class metadata.

## SLURM Usage Tips
- Package the commands above into your SLURM script (e.g., `sbatch train.sbatch`) after activating the Conda environment.
- Run all SLURM batch requests from the repository root so the relative paths resolve as documented.
- `--work-dir` defaults to `recognition/s4747674_YOLO`, keeping prepared data under version control boundaries.
- Training runs default to `recognition/s4747674_YOLO/runs`; inference runs default to `recognition/s4747674_YOLO/inference_runs`.
- Ultralytics checkpoints and plots are modest in size; periodically clean up these run directories when rerunning experiments.

### Provided Batch Scripts
- `recognition/s4747674_YOLO/sbatch_train.sh`: launches the full training job and logs to `recognition/s4747674_YOLO/logs/`.
- `recognition/s4747674_YOLO/sbatch_val.sh`: performs validation on the saved checkpoint.
- `recognition/s4747674_YOLO/sbatch_infer.sh`: runs inference over the prepared test split and saves annotated outputs.
Submit them with `sbatch recognition/s4747674_YOLO/sbatch_train.sh` (or the corresponding validation/inference script). Each script sources Conda, activates `isic-yolo`, and assumes the ISIC data live at `/home/groups/comp3710/ISIC2018`.
