"""Utility helpers for the ISIC lesion detection project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import regionprops


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(path: Path) -> Image.Image:
    """Load an RGB image from *path* using PIL."""

    with path.open("rb") as handle:
        img = Image.open(handle)
        return img.convert("RGB")


def load_mask(path: Path) -> np.ndarray:
    """Load a binary segmentation mask as a boolean numpy array."""

    with path.open("rb") as handle:
        mask = Image.open(handle)
        return np.array(mask) > 0


def mask_to_bboxes(mask: np.ndarray) -> List[np.ndarray]:
    """Convert a binary mask to a list of bounding boxes in ``xyxy`` order."""

    if mask.ndim != 2:
        raise ValueError("Segmentation mask must be a single channel 2D array")

    labeled = mask.astype(np.uint8)
    props = regionprops(labeled)
    boxes: List[np.ndarray] = []
    for prop in props:
        # regionprops supplies boxes as (row_min, col_min, row_max, col_max); convert to (x1, y1, x2, y2).
        min_row, min_col, max_row, max_col = prop.bbox
        boxes.append(np.array([min_col, min_row, max_col, max_row], dtype=np.float32))
    return boxes


def xyxy_to_yolo(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert bounding box from ``xyxy`` to YOLO ``xc yc w h`` normalised format."""

    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.array([xc / width, yc / height, w / width, h / height], dtype=np.float32)


def save_json(data: Mapping[str, object], path: Path) -> None:
    """Persist *data* as JSON to *path*."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


@dataclass
class PlotConfig:
    title: str
    x_label: str
    y_label: str


def plot_curves(
    curves: Mapping[str, Sequence[float]],
    output_path: Path,
    plot_config: PlotConfig,
) -> None:
    """Plot each curve in *curves* and save to *output_path*."""

    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    for label, values in curves.items():
        # Each metric or loss component is overlaid so the learning dynamics are directly comparable.
        plt.plot(values, label=label)
    plt.title(plot_config.title)
    plt.xlabel(plot_config.x_label)
    plt.ylabel(plot_config.y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_ultralytics_training_curves(results_csv: Path, output_dir: Path) -> Dict[str, Path]:
    """Create training curves from an Ultralytics ``results.csv`` artifact."""

    import pandas as pd

    df = pd.read_csv(results_csv)
    outputs: Dict[str, Path] = {}

    if {"train/box_loss", "train/cls_loss", "metrics/mAP50"}.issubset(df.columns):
        # Separate plots provide quick diagnostics for optimisation stability and detection quality.
        loss_curves = {
            "box_loss": df["train/box_loss"].tolist(),
            "cls_loss": df["train/cls_loss"].tolist(),
        }
        loss_path = output_dir / "loss_curves.png"
        plot_curves(
            loss_curves,
            loss_path,
            PlotConfig(
                title="Training Loss Components",
                x_label="Epoch",
                y_label="Loss",
            ),
        )
        outputs["loss"] = loss_path

        metrics_path = output_dir / "metrics.png"
        metrics_curves = {
            "mAP@0.5": df["metrics/mAP50"].tolist(),
        }
        if "metrics/mAP50-95" in df.columns:
            # Include the stricter COCO-style average when Ultralytics exports it.
            metrics_curves["mAP@0.5:0.95"] = df["metrics/mAP50-95"].tolist()
        plot_curves(
            metrics_curves,
            metrics_path,
            PlotConfig(
                title="Validation Metrics",
                x_label="Epoch",
                y_label="Score",
            ),
        )
        outputs["metrics"] = metrics_path

    return outputs
