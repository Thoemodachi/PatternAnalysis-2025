"""Training entry-point for ISIC lesion detection with YOLOv8."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .dataset import ISICDatasetConfig, ISICYoloDatasetBuilder
from .modules import YOLODetector
from .utils import plot_ultralytics_training_curves


def parse_args() -> argparse.Namespace:
    """Define and parse command line arguments for training and validation."""
    parser = argparse.ArgumentParser(description="Train YOLO detector on ISIC lesions")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to ISIC2018 root directory")
    default_work_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=default_work_dir,
        help="Directory for prepared data (default: recognition/s4747674_YOLO)",
    )
    default_project = (default_work_dir / "runs").resolve()
    parser.add_argument(
        "--project",
        type=Path,
        default=default_project,
        help="Ultralytics project directory for training runs",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model to start from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g. cuda, 0, cpu)")
    parser.add_argument("--force", action="store_true", help="Rebuild YOLO data even if it exists")
    parser.add_argument("--name", type=str, default="yolov8n_isic", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume latest run if available")
    parser.add_argument("--val-only", action="store_true", help="Skip training and run validation only")
    return parser.parse_args()


def prepare_dataset(data_root: Path, work_dir: Path, force: bool = False) -> Path:
    """Build (or reuse) the YOLO-formatted dataset and return the YAML manifest."""
    config = ISICDatasetConfig(root=data_root)
    builder = ISICYoloDatasetBuilder(config, work_dir)
    return builder.prepare(force=force)


def train_detector(
    data_yaml: Path,
    model_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: Path,
    name: str,
    device: Optional[str] = None,
    resume: bool = False,
) -> Path:
    """Execute Ultralytics training and return the run directory containing artefacts."""
    detector = YOLODetector(model_path=model_path, device=device)
    if resume:
        # Resume is handled internally by Ultralytics; ensure paths exist_ok.
        detector.train(
            resume=True,
            data=str(data_yaml),
            project=str(project),
            name=name,
            exist_ok=True,
        )
    else:
        # Fresh training run following the requested hyperparameters.
        detector.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(project),
            name=name,
            exist_ok=True,
        )
    trainer = getattr(detector.model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Trainer state missing after training. Check Ultralytics version.")
    return Path(trainer.save_dir)


def validate_detector(
    data_yaml: Path,
    model_path: str,
    project: Path,
    name: str,
    device: Optional[str] = None,
) -> None:
    """Evaluate a trained checkpoint on the validation split."""
    detector = YOLODetector(model_path=model_path, device=device)
    # Persist validation results in their own sub-run to avoid overwriting training artefacts.
    detector.val(data=str(data_yaml), project=str(project), name=f"{name}_val", split="val")


def main() -> None:
    args = parse_args()
    # Materialise YOLO-formatted assets so Ultralytics can ingest the dataset.
    data_yaml = prepare_dataset(args.data_root, args.work_dir, force=args.force)

    if args.val_only:
        # Validation-only mode attaches to an existing checkpoint and skips training.
        validate_detector(data_yaml, args.model, args.project, args.name, args.device)
        return

    run_dir = train_detector(
        data_yaml=data_yaml,
        model_path=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        resume=args.resume,
    )

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        # Generate diagnostic plots (loss, mAP) to streamline reporting.
        plot_ultralytics_training_curves(results_csv, run_dir)


if __name__ == "__main__":
    main()
