# ISIC Lesion Detection (YOLOv8)
This project trains a YOLOv8 object detector on the ISIC 2018 dermoscopic imaging challenge to localise skin lesions with high IoU (>0.8) and classify lesion type. The pipeline converts the official segmentation masks into bounding boxes, builds a YOLO-compatible dataset, trains a detector, plots training dynamics, and runs inference on new images.

## Project Layout
- `modules.py`: thin wrapper around Ultralytics YOLO models to keep training, validation, and inference logic reusable.
- `dataset.py`: dataset configuration and builder that transforms segmentation masks into bounding boxes and prepares the YOLO dataset structure.
- `utils.py`: shared helpers for loading masks/images, converting boxes, and plotting training curves.
- `train.py`: end-to-end training script that materialises the dataset, runs YOLOv8 training, and saves diagnostic plots.
- `predict.py`: command line inference driver for visualising predictions on arbitrary inputs.

## Prerequisites
1. Download the ISIC 2018 challenge data to a location accessible from the Rangpur cluster (e.g. `/home/groups/comp3710/ISIC2018`). Ensure the Task 1/2 images, segmentation masks, and Task 3 ground-truth CSV are present.
2. Create a Python environment with PyTorch (CUDA build recommended) and install the required packages:
   ```bash
   pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install pandas matplotlib scikit-image pyyaml
   ```

## Dataset Preparation & Training
Prepare the YOLO dataset and launch training (adjust paths, epochs, and batch size for the cluster resources):
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
The script will:
- materialise `prepared/` under `--work-dir` with YOLO-formatted images/labels and an `isic.yaml` manifest;
- run Ultralytics training with the requested hyperparameters (resumeable via `--resume`);
- emit plots of loss curves and validation metrics in the Ultralytics run directory.

## Validation
To evaluate a trained checkpoint without retraining:
```bash
python -m recognition.s4747674_YOLO.train \
  --data-root /home/groups/comp3710/ISIC2018 \
  --work-dir /scratch/$USER/isic-yolo \
  --model runs/isic/yolov8m_isic/weights/best.pt \
  --val-only
```

## Inference
Generate predictions for a folder of dermoscopic images:
```bash
python -m recognition.s4747674_YOLO.predict \
  --model runs/isic/yolov8m_isic/weights/best.pt \
  --source /scratch/$USER/isic-yolo/prepared/images/test \
  --save
```
Annotated outputs and detection summaries will be written under `runs/isic_pred/` by default.

## Notes
- Bounding boxes are derived from segmentation masks; adjust `ISICDatasetConfig.min_mask_area` if tiny masks should be ignored or retained.
- The code keeps image copies via symbolic links where supported. On filesystems without symlink support, images are duplicated instead.
- Extend `utils.plot_ultralytics_training_curves` to track additional metrics (e.g. classification accuracy) if needed for reporting.
- Use the generated `results.csv`, loss plots, and Ultralytics TensorBoard logs for the required graphs and analysis in your report.
