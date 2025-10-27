# ISIC Lesion Detection with YOLOv8
This project targets automated localisation of lesions within the ISIC 2018 dermoscopic benchmark. A pre-trained YOLOv8 detector is adapted to the binary lesion detection task, ensuring detections exceed an Intersection over Union (IoU) threshold of 0.8 on the test set while sustaining reliable classification behaviour.  The pipeline prioritises clinician-friendly interpretability by translating pixel-wise masks into bounding boxes that highlight lesion extent.

## Method Overview
The algorithm converts the official ISIC segmentation masks into YOLO-formatted bounding boxes, constructs a reproducible train/validation/test split, and fine-tunes YOLOv8 using Ultralytics’ training API. The training routine monitors mAP@0.5 and mAP@0.5:0.95 to quantify detection fidelity. Inference reuses the same wrapper to produce per-image detections and qualitative overlays for reporting.

![TODO: Qualitative detection overlay](TODO)

## Dependencies and Environment
- Python 3.11
- Ultralytics 8.1.x
- PyTorch 2.1.x with CUDA 11.8
- torchvision 0.16.x
- torchaudio 2.1.x
- pandas 2.1.x
- matplotlib 3.8.x
- scikit-image 0.22.x
- PyYAML 6.0.x
- Pillow 10.x

Install the environment via:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas matplotlib scikit-image pyyaml pillow
```

## Data Preparation and Pre-processing
1. Download the ISIC 2018 challenge imagery and segmentation masks to a shared location (e.g. `/home/groups/comp3710/ISIC2018`). The dataset is described by Codella et al. [1].
2. `ISICYoloDatasetBuilder` links image assets into `prepared/images/{train,val,test}` and writes YOLO labels into `prepared/labels/{train,val,test}`.
3. Segmentation masks are thresholded to binary and converted to bounding boxes using `skimage.measure.regionprops`, retaining the minimum-area rectangle around each connected component.
4. Masks with fewer than 20 foreground pixels are discarded to remove artefacts, reflecting the recommendation in the ISIC metadata to ignore tiny annotation fragments [1].

The dataset is split 70 %/15 %/15 % (train/validation/test). This mirrors the split used in comparable ISIC detection studies, ensuring sufficient validation coverage for hyper-parameter selection while preserving a representative test set for final reporting. The split is seeded to guarantee deterministic behaviour when rebuilding the dataset.

## Training and Inference Workflow
Prepare the dataset and launch fine-tuning:
```bash
python -m recognition.s4747674_YOLO.train \
  --data-root /home/groups/comp3710/ISIC2018 \
  --work-dir recognition/s4747674_YOLO \
  --model yolov8m.pt \
  --epochs 100 \
  --batch 32 \
  --imgsz 640 \
  --project recognition/s4747674_YOLO/runs \
  --name yolov8m_isic
```

Validate a trained checkpoint:
```bash
python -m recognition.s4747674_YOLO.train \
  --data-root /home/groups/comp3710/ISIC2018 \
  --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt \
  --val-only
```

Generate qualitative predictions:
```bash
python -m recognition.s4747674_YOLO.predict \
  --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt \
  --source recognition/s4747674_YOLO/prepared/images/test \
  --save \
  --project recognition/s4747674_YOLO/inference_runs \
  --name yolov8m_isic_test
```

## Outputs and Logging
- Training artefacts reside in `recognition/s4747674_YOLO/runs/<run_name>`:
  - `weights/best.pt` and `weights/last.pt` checkpoints.
  - `results.csv` containing per-epoch loss and IoU-centred metrics (mAP@0.5, mAP@0.5:0.95).
  - `loss_curves.png` and `metrics.png` illustrating optimisation and validation behaviour.
- Validation and inference logs are collected in `recognition/s4747674_YOLO/logs/` via SLURM batch scripts.
- Qualitative overlays are written to `recognition/s4747674_YOLO/inference_runs/<name>`.

Example terminal output (abbreviated):
```text
image_0001.jpg: class=0 score=0.91 bbox=[134.2, 98.7, 256.4, 210.6]
image_0002.jpg: no detections
```

Example plot expectations:
- TODO: Training loss curve showing convergence over 100 epochs.
- TODO: Validation mAP curves demonstrating IoU≥0.8 performance.

## Reproducibility Notes
1. Seeded data splitting (`seed=13`) ensures stable partitioning across runs.
2. Ultralytics training is deterministic when `torch.backends.cudnn.deterministic=True`; set `CUBLAS_WORKSPACE_CONFIG=:16:8`.
3. Record the `ultralytics` package version and Git commit hash of this repository in experiment logs.
4. Use SLURM scripts (`sbatch_train.sh`, `sbatch_val.sh`, `sbatch_infer.sh`) to standardise cluster execution; they activate the `isic-yolo` Conda environment to replicate results.

## Known TODOs
- TODO: Insert qualitative figure illustrating detections on the ISIC test split.
- TODO: Populate final test-set IoU≥0.8 statistics (precision, recall, mAP).
- TODO: Add confusion matrix for lesion classification if multi-class labels are incorporated.

## References
[1] Codella, N. C. F., Rotemberg, V., Tschandl, P., et al. (2019). Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC). *arXiv:1902.03368*.
