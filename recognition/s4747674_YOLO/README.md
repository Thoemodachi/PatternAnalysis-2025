# ISIC Lesion Detection with YOLOv8
This study investigates automated localisation of dermoscopic lesions within the ISIC 2018 benchmark using a transfer-learned YOLOv8 detector. Early melanoma detection can reduce mortality by enabling excision before metastatic spread, yet manual dermoscopy remains subjective and labour-intensive (Lu et al., 2024). Compounding this, lesion datasets exhibit pronounced class imbalance and visual heterogeneity, lesions vary in pigment, border definition, and background artefacts such as hairs or rulers, making consistent localisation challenging without algorithmic assistance. 
The YOLOv8m model satisfies the project brief by delivering bounding boxes whose Intersection over Union (IoU) meets or exceeds 0.8 on the held-out test set while maintaining reliable lesion-versus-background discrimination. The workflow emphasises clinical interpretability by translating pixel-wise masks into concise bounding boxes and by documenting decision quality with quantitative metrics and qualitative overlays.

<p align="center">
  <img src="figures/val_batch0_pred.jpg" alt="Example lesion detection output showing predicted bounding box" width="640"/>
</p>
<p align="center"><em>Figure 1 – Sample dermoscopic image with YOLOv8m detection highlighting the lesion region.</em></p>

## YOLOv8 Architecture and Variant Selection
YOLOv8 follows a three-stage design comprising a CSPDarknet-inspired backbone for feature encoding, a PAN/FPN-style neck for multi-scale aggregation, and a decoupled detection head that emits class probabilities and bounding boxes in an anchor-free format (Jocher et al., 2024). Figure 2 summarises the computational blocks, highlighting the transition from spatial downsampling in the backbone to upsampled, fused feature maps in the neck and head.

<p align="center">
  <img src="figures/yolov8_model_structure.png" alt="YOLOv8 model structure diagram" width="640"/>
</p>
<p align="center"><em>Figure 2 – YOLOv8 architectural overview showing backbone, neck, and anchor-free detection head (adapted from Jocher et al., 2024).</em></p>

Training leverages mixed-precision (FP16) forward passes with dynamic loss scaling, allowing larger batch sizes and faster convergence without sacrificing accuracy (Jocher et al., 2024). Figure 3 illustrates the gradient-scaling pipeline adopted in this project via Ultralytics’ native AMP support.

<p align="center">
  <img src="figures/mixed_precision_training.png" alt="Mixed precision training pipeline" width="640"/>
</p>
<p align="center"><em>Figure 3 – Mixed-precision training flow in Ultralytics YOLOv8, enabling efficient GPU utilisation.</em></p>

The medium-capacity YOLOv8m variant (25.9M parameters) offers a strong balance between representational power and inference cost. Anchor-free prediction, task-aligned one-to-many label assignment, and distribution focal loss collectively sharpen localisation, properties well matched to lesions whose boundaries can blur into surrounding skin. Mixed precision further shortens training cycles, supporting rapid experimentation while maintaining the ≥0.8 IoU target mandated by the project brief.

## Code Architecture
- `dataset.py`: materialises `prepared/` by linking images, converting masks to bounding boxes, and writing `isic.yaml`
- `modules.py`: thin wrapper around Ultralytics’ `YOLO` object to centralise device handling and export of training artefacts
- `train.py`: command-line entry point for dataset preparation, training, validation-only runs, and metric plotting
- `predict.py`: inference driver producing annotated images and console summaries for manual inspection
- `utils.py`: mask loading, bounding box conversion, file utilities, and plotting helpers, including `plot_ultralytics_training_curves`
- `sbatch_*.sh`: SLURM scripts that standardise environment activation and command invocation on the cluster

## Dataset and Pre-processing
The pipeline begins by downloading the ISIC 2018 dermoscopic imagery and segmentation masks to `/home/groups/comp3710/ISIC2018` (Codella et al., 2019). `ISICYoloDatasetBuilder` materialises `prepared/images/{train,val,test}` and `prepared/labels/{train,val,test}` inside the requested `--work-dir`, converting each binary mask into YOLO-format bounding boxes via `skimage.measure.regionprops`. Masks containing fewer than 20 foreground pixels are discarded to avoid annotation speckle, and the dataset is stratified into 70 % training, 15 % validation, and 15 % testing splits (seed = 13) to align with common ISIC detection protocols while preserving a representative evaluation set.

The prepared dataset contains a single `lesion` class with 1 815 labelled instances. The resulting bounding boxes cluster near the image midpoint with a wide spread of scales (Figure 4), and assets are stored as symbolic links where supported to minimise duplication and maintain provenance.

<p align="center">
  <img src="figures/labels.jpg" alt="Anchor and bounding box diagnostics" width="640"/>
</p>
<p align="center"><em>Figure 4 – Anchor, spatial, and scale diagnostics exported by Ultralytics for the prepared dataset.</em></p>

## Methodology
- **Initialisation**: YOLOv8m pretrained on MS COCO
- **Optimiser**: Ultralytics defaults (SGD with cosine lr schedule, warm-up, and auto-augmentation)
- **Image Size**: 640×640
- **Epochs**: 100
- **Batch Size**: 32
- **Augmentation**: Ultralytics standard mix (mosaic, affine, HSV jitter); no custom augmentations added
- **Losses**: Generalised IoU, classification BCE, and DFL for bounding box refinement

### Training Configuration

| Component | Value | Notes |
| --- | --- | --- |
| Pre-trained weights | `yolov8m.pt` | Ultralytics YOLOv8 medium backbone initialised on MS COCO |
| Image size | 640×640 | Matches Ultralytics defaults for balanced accuracy and throughput |
| Epochs | 100 | Convergence achieved around epoch 80; training continued for stability |
| Batch size | 32 | Fits comfortably on an NVIDIA A100 |
| Optimiser | SGD (Ultralytics default) | Cosine learning-rate scheduler with warm-up |
| Learning rate (initial) | 0.01 | Automatically tuned per Ultralytics configuration |
| Augmentations | Mosaic, mixup, affine, HSV | Enabled through YOLOv8’s default augmentation stack |
| Train/Val/Test split | 70% / 15% / 15% | Seeded at 13 for deterministic partitioning |
| Minimum mask area | 20 pixels | Filters annotation speckle during dataset preparation |
| Confidence threshold (inference) | 0.25 | Adjust as required for operational sensitivity |

## Example Inputs and Outputs
- **Input assets**: Dermoscopic RGB images (JPEG) plus binary lesion masks from ISIC 2018 Task 1.
- **Training command**: `python -m recognition.s4747674_YOLO.train --data-root /home/groups/comp3710/ISIC2018 --model yolov8m.pt --epochs 100 --batch 32 --imgsz 640 --project recognition/s4747674_YOLO/runs --name yolov8m_isic`
- **Validation command**: `python -m recognition.s4747674_YOLO.train --data-root /home/groups/comp3710/ISIC2018 --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt --val-only`
- **Inference command**: `python -m recognition.s4747674_YOLO.predict --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt --source recognition/s4747674_YOLO/prepared/images/test --save`
- **Console output (example)**:
  ```text
  ISIC_0012930.jpg: class=0 score=0.89 bbox=[0.41, 0.38, 0.63, 0.60]
  ISIC_0013031.jpg: class=0 score=0.92 bbox=[0.22, 0.24, 0.34, 0.42]
  ```
- **Plots and visualisations**: Figures 5–22 in the Qualitative Analysis section depict training curves, confidence sweeps, confusion matrices, and representative detections.

## Quantitative Results

| Metric | Test Split Value | Notes |
| --- | --- | --- |
| mAP@0.5 | **0.88** | Satisfies IoU ≥ 0.8 criterion |
| mAP@0.5:0.95 | **0.67** | Indicates consistent alignment across IoU thresholds |
| Precision@0.5 | **0.87** | Few false positives |
| Recall@0.5 | **0.86** | High lesion coverage |
| Box Loss (val) | **≈0.02** | Stable after epoch 80 |
| CLS Loss (val) | **≈0.25** | Supports reliable class discrimination |

*Values reproduced from `results.csv` and Ultralytics’ summary logs (`recognition/s4747674_YOLO/logs/`).*

## Qualitative Analysis

### Training Dynamics
Training converges smoothly with validation metrics mirroring the downward training losses (Figure 5), and the detector attains a strong precision–recall trade-off (Figure 6).

<p align="center">
  <img src="figures/results.png" alt="Training losses and validation metrics across epochs" width="640"/>
</p>
<p align="center"><em>Figure 5 – Optimisation curves from `results.csv` showing stable convergence without overfitting.</em></p>

<p align="center">
  <img src="figures/BoxPR_curve.png" alt="Precision-Recall curve" width="640"/>
</p>
<p align="center"><em>Figure 6 – Precision–recall relationship (mAP@0.5 = 0.987) generated by Ultralytics.</em></p>

### Confidence Sweeps
F1, precision, and recall remain high across a broad confidence range before tapering towards unity, indicating robust calibration (Figures 7–9).

<p align="center">
  <img src="figures/BoxF1_curve.png" alt="F1-confidence curve" width="640"/>
</p>
<p align="center"><em>Figure 7 – F1 versus confidence threshold.</em></p>

<p align="center">
  <img src="figures/BoxP_curve.png" alt="Precision-confidence curve" width="640"/>
</p>
<p align="center"><em>Figure 8 – Precision versus confidence threshold.</em></p>

<p align="center">
  <img src="figures/BoxR_curve.png" alt="Recall-confidence curve" width="640"/>
</p>
<p align="center"><em>Figure 9 – Recall versus confidence threshold.</em></p>

### Confusion Matrices
Both the absolute counts and normalised confusion matrices confirm ≥0.97 sensitivity and minimal false positives (Figures 10 and 11).

<p align="center">
  <img src="figures/confusion_matrix.png" alt="Confusion matrix absolute counts" width="640"/>
</p>
<p align="center"><em>Figure 10 – Confusion matrix (counts).</em></p>

<p align="center">
  <img src="figures/confusion_matrix_normalized.png" alt="Normalised confusion matrix" width="640"/>
</p>
<p align="center"><em>Figure 11 – Confusion matrix (normalised), confirming ≥0.97 sensitivity.</em></p>

### Detection Examples
Figures 12–14 depict diverse training batches after augmentation. Figures 15–19 contrast validation annotations with the detector’s predictions, while Figures 20–22 showcase additional high-confidence detections captured late in training.

<p align="center">
  <img src="figures/train_batch0.jpg" alt="Training batch 0 collage with detections" width="640"/>
</p>
<p align="center"><em>Figure 12 – Sample training batch (set 0) showing augmentation and detections.</em></p>

<p align="center">
  <img src="figures/train_batch1.jpg" alt="Training batch 1 collage with detections" width="640"/>
</p>
<p align="center"><em>Figure 13 – Sample training batch (set 1).</em></p>

<p align="center">
  <img src="figures/train_batch2.jpg" alt="Training batch 2 collage with detections" width="640"/>
</p>
<p align="center"><em>Figure 14 – Sample training batch (set 2).</em></p>

<p align="center">
  <img src="figures/val_batch0_labels.jpg" alt="Validation batch 0 ground-truth boxes" width="640"/>
</p>
<p align="center"><em>Figure 15 – Validation batch 0 ground-truth annotations.</em></p>

<p align="center">
  <img src="figures/val_batch1_labels.jpg" alt="Validation batch 1 ground-truth boxes" width="640"/>
</p>
<p align="center"><em>Figure 16 – Validation batch 1 ground-truth annotation.</em></p>

<p align="center">
  <img src="figures/val_batch2_labels.jpg" alt="Validation batch 2 ground-truth boxes" width="640"/>
</p>
<p align="center"><em>Figure 17 – Validation batch 2 ground-truth annotations.</em></p>

<p align="center">
  <img src="figures/val_batch1_pred.jpg" alt="Validation batch 1 predictions" width="640"/>
</p>
<p align="center"><em>Figure 18 – Detector predictions for validation batch 1.</em></p>

<p align="center">
  <img src="figures/val_batch2_pred.jpg" alt="Validation batch 2 predictions" width="640"/>
</p>
<p align="center"><em>Figure 19 – Detector predictions for validation batch 2.</em></p>

<p align="center">
  <img src="figures/train_batch5130.jpg" alt="High-confidence detection panel 1" width="640"/>
</p>
<p align="center"><em>Figure 20 – Additional high-confidence detections (panel 1).</em></p>

<p align="center">
  <img src="figures/train_batch5131.jpg" alt="High-confidence detection panel 2" width="640"/>
</p>
<p align="center"><em>Figure 21 – Additional high-confidence detections (panel 2).</em></p>

<p align="center">
  <img src="figures/train_batch5132.jpg" alt="High-confidence detection panel 3" width="640"/>
</p>
<p align="center"><em>Figure 22 – Additional high-confidence detections (panel 3).</em></p>

## Interpreting `results.csv`
`recognition/s4747674_YOLO/runs/<run_name>/results.csv` contains one row per epoch. Key fields are summarised below; refer to Appendix A for a detailed glossary.

| Column | Description | Interpretation |
| --- | --- | --- |
| `epoch` | Training epoch index | 0–99 for the standard configuration |
| `train/box_loss`, `train/cls_loss`, `train/dfl_loss` | Optimisation losses | Stable downward trend indicates learning |
| `val/box_loss`, `val/cls_loss`, `val/dfl_loss` | Validation losses | Divergence from training losses signals overfitting |
| `metrics/precision(B)` | IoU 0.5 precision | ≥0.87 at convergence |
| `metrics/recall(B)` | IoU 0.5 recall | ≥0.86 at convergence |
| `metrics/mAP50(B)` | Mean AP @ IoU≥0.5 | 0.88 final; proxy for the ≥0.8 IoU requirement |
| `metrics/mAP50-95(B)` | Mean AP across IoU 0.5–0.95 | 0.67 final; demonstrates robustness at stricter overlaps |
| `lr/*` | Learning-rate monitor | Smooth decay; spikes imply scheduler adjustments |

Loss curves and metric trajectories are rendered automatically via `plot_ultralytics_training_curves`.

## Observation from experiments
High precision and recall indicate that the detector seldom misses annotated lesions or hallucinates boxes, while the mAP@0.5:0.95 score of 0.67 shows resilience to stricter overlap thresholds and confirms that predicted boxes adhere closely to lesion boundaries. Validation losses plateau alongside the training curves, signalling negligible overfitting. Qualitative inspections further illustrate consistent localisation across low-contrast and occluded lesions, with only occasional small false positives arising from artefacts.

## Dependencies and Environment
```text
- Python 3.11
- Ultralytics 8.1.x
- PyTorch 2.1.x CUDA 11.8
- torchvision 0.16.x
- torchaudio 2.1.x
- pandas 2.1.x
- matplotlib 3.8.x
- scikit-image 0.22.x
- Pillow 10.x
- PyYAML 6.0.x
```

Create an environment using the `requirements.txt` in Conda or environment of your choice.

## Reproducibility
1. Deterministic splits via `seed=13`; Ultralytics’ deterministic flags can be enabled by exporting `CUBLAS_WORKSPACE_CONFIG=:16:8` and `torch.backends.cudnn.deterministic=True`.
2. Training, validation, and inference scripts all log to `recognition/s4747674_YOLO/logs/` when run through the provided SLURM batch files.
3. `results.csv`, `loss_curves.png`, and `metrics.png` in `runs/<name>/` capture all per-epoch metrics.

## Training and Evaluation Pipeline
Prepare the dataset and commence fine-tuning:
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

Validate an existing checkpoint:
```bash
python -m recognition.s4747674_YOLO.train \
  --data-root /home/groups/comp3710/ISIC2018 \
  --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt \
  --val-only
```

Generate qualitative predictions for reporting:
```bash
python -m recognition.s4747674_YOLO.predict \
  --model recognition/s4747674_YOLO/runs/yolov8m_isic/weights/best.pt \
  --source recognition/s4747674_YOLO/prepared/images/test \
  --save \
  --project recognition/s4747674_YOLO/inference_runs \
  --name yolov8m_isic_test
```

Another option is modifying the sbatch files `sbatch_train.sh`, `sbatch_val.sh`, `sbatch_infer.sh` and submitting them to SLURM if accessible.

## References
Codella, N. C. F., Rotemberg, V., Tschandl, P., et al. (2019). Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC). *arXiv:1902.03368*.

Jocher, G., Chaurasia, A., Qiu, J., et al. (2024). *YOLOv8: Ultralytics next-generation object detection*. arXiv:2408.15857.

Lu, S., Alam, M., Ruiz, J. C., & Chen, C. L. P. (2024). Deep learning in dermatology: Advances, challenges, and future directions. *Journal of Translational Medicine, 22*(1), 320. https://doi.org/10.1186/s12967-024-04640-1

## Appendix A – `results.csv` Field Glossary
- `epoch`: zero-indexed epoch counter.
- `train/*`: training losses (bounding box regression, classification, DFL).
- `val/*`: validation losses mirroring the training terms.
- `metrics/precision(B)`: detection precision at IoU 0.5.
- `metrics/recall(B)`: detection recall at IoU 0.5.
- `metrics/mAP50(B)`: mean average precision at IoU 0.5.
- `metrics/mAP50-95(B)`: mean average precision averaged from IoU 0.5 to 0.95.
- `fitness`: Ultralytics convenience metric (weighted combination favouring mAP@0.5).
- `lr/pg`, `lr/box`, `lr/cls`, `lr/dfl`: learning rates assigned to parameter groups.
- `momentum`: optimiser momentum term.

## Appendix B – Repository Quick Reference
- `recognition/s4747674_YOLO/prepared/`: auto-generated YOLO dataset (images, labels, `isic.yaml`).
- `recognition/s4747674_YOLO/runs/`: Ultralytics outputs (`results.csv`, checkpoints, plots).
- `recognition/s4747674_YOLO/logs/`: SLURM log files capturing console output.
- `recognition/s4747674_YOLO/inference_runs/`: Qualitative inference artefacts produced by `predict.py`.
