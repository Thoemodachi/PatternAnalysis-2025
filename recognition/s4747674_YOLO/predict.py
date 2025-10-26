"""Inference helper for the trained ISIC YOLO detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from .modules import YOLODetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained weights or model name")
    parser.add_argument("--source", type=str, required=True, help="Image file, directory, or glob pattern")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true", help="Persist annotated predictions")
    default_project = (Path(__file__).resolve().parent / "inference_runs").resolve()
    parser.add_argument(
        "--project",
        type=Path,
        default=default_project,
        help="Output directory for inference runs",
    )
    parser.add_argument("--name", type=str, default="inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YOLODetector(model_path=args.model, device=args.device)
    predictions = detector.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    for result in predictions:
        path = Path(result.path)
        if getattr(result, "boxes", None) is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
        else:
            boxes, scores, classes = [], [], []
        summary = ", ".join(
            f"class={int(cls)} score={float(score):.2f} bbox={bbox}"
            for cls, score, bbox in zip(classes, scores, boxes)
        )
        print(f"{path.name}: {summary or 'no detections'}")


if __name__ == "__main__":
    main()
