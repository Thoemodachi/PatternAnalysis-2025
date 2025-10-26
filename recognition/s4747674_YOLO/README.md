# ISIC Lesion Detection Workflow
This package trains and evaluates a YOLOv8 detector on the ISIC 2018 dermoscopy dataset. It converts the segmentation masks into YOLO-ready bounding boxes, launches Ultralytics training, and runs inference for qualitative checks.

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
  --work-dir /scratch/$USER/isic-yolo \
  --model yolov8m.pt \
  --epochs 100 \
  --batch 32 \
  --imgsz 640 \
  --project runs/isic \
  --name yolov8m_isic
```
This command will
- build `prepared/` inside `--work-dir` with YOLO-formatted images, labels, and `isic.yaml`;
- train the YOLO model (resumable via `--resume`);
- generate `results.csv` plus loss/metric plots in the Ultralytics run directory.

## Validation Only
```bash
python -m recognition.s4747674_YOLO.train \
  --data-root /home/groups/comp3710/ISIC2018 \
  --work-dir /scratch/$USER/isic-yolo \
  --model runs/isic/yolov8m_isic/weights/best.pt \
  --val-only
```

## Inference & Visualisation
```bash
python -m recognition.s4747674_YOLO.predict \
  --model runs/isic/yolov8m_isic/weights/best.pt \
  --source /scratch/$USER/isic-yolo/prepared/images/test \
  --save
```
Predictions are written to `runs/isic_pred/` (configurable via `--project`/`--name`).

## Helpful Flags
- `--force`: rebuild the YOLO dataset even if it already exists.
- `--device cuda:0`: choose GPU or CPU explicitly.
- `--conf 0.4`: adjust confidence threshold when running inference.
- `--export`: available via `YOLODetector.export` if you need ONNX/TensorRT exports.

