"""Model components for ISIC lesion detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _load_ultralytics() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as error:
        message = (
            "ultralytics package is required. Install it via `pip install ultralytics` "
            "before running training or inference."
        )
        raise ModuleNotFoundError(message) from error
    return YOLO


@dataclass
class YOLODetector:
    """Thin wrapper around Ultralytics YOLO models."""

    model_path: str = "yolov8n.pt"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        YOLO = _load_ultralytics()
        self.model = YOLO(self.model_path)

    def train(self, **kwargs: Any) -> Any:
        """Train the detector with keyword arguments forwarded to Ultralytics."""

        if self.device is not None:
            kwargs.setdefault("device", self.device)
        return self.model.train(**kwargs)

    def val(self, **kwargs: Any) -> Any:
        if self.device is not None:
            kwargs.setdefault("device", self.device)
        return self.model.val(**kwargs)

    def predict(self, **kwargs: Any) -> Any:
        if self.device is not None:
            kwargs.setdefault("device", self.device)
        return self.model.predict(**kwargs)

    def export(self, **kwargs: Any) -> Any:
        return self.model.export(**kwargs)

    def save_training_artifacts(self, run_dir: Path) -> Dict[str, Path]:
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            raise FileNotFoundError(
                f"results.csv not found in {run_dir}. Ensure Ultralytics training completed successfully"
            )
        return {"results_csv": results_csv}

