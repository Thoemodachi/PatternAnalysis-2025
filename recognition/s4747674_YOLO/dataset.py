from __future__ import annotations

import random
import shutil
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from . import utils

DEFAULT_CLASSES: Sequence[str] = ("lesion",)

@dataclass
class ISICDatasetConfig:
    """Configuration describing where to find the ISIC2018 assets."""

    root: Path
    images_dir: Path = field(init=False)
    masks_dir: Path = field(init=False)
    class_names: Sequence[str] = field(default_factory=lambda: DEFAULT_CLASSES)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 13
    min_mask_area: int = 20

    def __post_init__(self) -> None:
        """Resolve dataset directories and validate configured split ratios."""
        self.root = Path(self.root)
        self.images_dir = self._resolve_dir(
            [
                self.root / "ISIC2018_Task1-2_Training_Input",
                self.root / "ISIC2018_Task1-2_Training_Input_x2",
            ]
        )
        self.masks_dir = self._resolve_dir(
            [
                self.root / "ISIC2018_Task1_Training_GroundTruth",
                self.root / "ISIC2018_Task1_Training_GroundTruth_x2",
            ]
        )
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("train/val/test ratios must sum to 1.0")

    @staticmethod
    def _resolve_dir(candidates: Sequence[Path]) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "None of the directory candidates exist: "
            + ", ".join(str(path) for path in candidates)
        )

@dataclass
class ISICSample:
    """Lightweight record tying an image to its mask and YOLO class index."""
    image_id: str
    image_path: Path
    mask_path: Path
    class_index: int


class ISICYoloDatasetBuilder:
    """Convert ISIC data with segmentation masks into YOLO detection format."""

    def __init__(self, config: ISICDatasetConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = Path(work_dir)
        self.dataset_root = self.work_dir / "prepared"

    def prepare(self, force: bool = False) -> Path:
        """Prepare the YOLO dataset and return the YAML manifest path."""

        yaml_path = self.dataset_root / "isic.yaml"
        if yaml_path.exists() and not force:
            # Reuse previously prepared assets unless a rebuild is explicitly requested
            return yaml_path

        utils.ensure_dir(self.dataset_root)
        samples = self._collect_samples()
        splits = self._split_samples(samples)
        image_root = self.dataset_root / "images"
        label_root = self.dataset_root / "labels"
        for split, subset in splits.items():
            self._write_split(split, subset, image_root, label_root)
        self._write_dataset_yaml(yaml_path, image_root)
        return yaml_path

    def _collect_samples(self) -> List[ISICSample]:
        """Gather dataset samples, filtering out entries without valid masks."""
        samples = self._collect_from_files()
        if not samples:
            raise RuntimeError(
                "No valid ISIC samples discovered. Check dataset paths and mask naming."
            )

        return samples

    def _collect_from_files(self) -> List[ISICSample]:
        """Walk the image directory to match each image with its segmentation mask."""
        samples: List[ISICSample] = []
        for image_path in sorted(self.config.images_dir.glob("*.jpg")):
            image_id = image_path.stem
            mask_path = self._find_mask(image_id)
            if mask_path is None:
                continue
            mask = utils.load_mask(mask_path)
            if mask.sum() < self.config.min_mask_area:
                # Ignore samples where the segmentation is effectively empty noise
                continue
            samples.append(
                ISICSample(
                    image_id=image_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    class_index=0,
                )
            )
        for image_path in sorted(self.config.images_dir.glob("*.png")):
            # Some sites provide PNG assets; process them in the same manner
            image_id = image_path.stem
            mask_path = self._find_mask(image_id)
            if mask_path is None:
                continue
            mask = utils.load_mask(mask_path)
            if mask.sum() < self.config.min_mask_area:
                # Ignore samples where the segmentation is effectively empty noise
                continue
            samples.append(
                ISICSample(
                    image_id=image_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    class_index=0,
                )
            )
        return samples

    def _find_mask(self, image_id: str) -> Optional[Path]:
        """Identify the matching segmentation mask file for a given image identifier."""
        candidates = [
            self.config.masks_dir / f"{image_id}_segmentation.png",
            self.config.masks_dir / f"{image_id}_segmentation.jpg",
            self.config.masks_dir / f"{image_id}.png",
            self.config.masks_dir / f"{image_id}.jpg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _split_samples(self, samples: Sequence[ISICSample]) -> Dict[str, List[ISICSample]]:
        """Shuffle and partition samples according to the configured ratios."""
        rng = random.Random(self.config.seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        # Deterministic boundaries ensure downstream reproducibility across runs
        total = len(shuffled)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)
        train_split = shuffled[:train_end]
        val_split = shuffled[train_end:val_end]
        test_split = shuffled[val_end:]
        return {"train": train_split, "val": val_split, "test": test_split}

    def _write_split(
        self,
        split: str,
        samples: Sequence[ISICSample],
        image_root: Path,
        label_root: Path,
    ) -> None:
        """Populate the YOLO directory structure for a dataset split."""
        split_image_dir = utils.ensure_dir(image_root / split)
        split_label_dir = utils.ensure_dir(label_root / split)
        for sample in samples:
            target_image = split_image_dir / sample.image_path.name
            if not target_image.exists():
                # Prefer symbolic links to minimise storage overhead when the filesystem permits
                self._link_or_copy(sample.image_path, target_image)

            label_path = split_label_dir / f"{sample.image_id}.txt"
            self._write_label(label_path, sample)

    def _write_label(self, label_path: Path, sample: ISICSample) -> None:
        """Convert segmentation masks into YOLO-compatible bounding box annotations."""
        image = utils.load_image(sample.image_path)
        width, height = image.size
        mask = utils.load_mask(sample.mask_path)
        boxes = utils.mask_to_bboxes(mask)
        if not boxes:
            label_path.write_text("", encoding="utf-8")
            return
        yolo_boxes = [utils.xyxy_to_yolo(box, width, height) for box in boxes]
        lines = [
            f"{sample.class_index} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
            for bbox in yolo_boxes
        ]
        label_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_dataset_yaml(self, yaml_path: Path, image_root: Path) -> None:
        """Save the Ultralytics dataset configuration so runs remain relocatable."""
        import yaml

        data = {
            "path": str(self.dataset_root.resolve()),
            "train": str((image_root / "train").resolve()),
            "val": str((image_root / "val").resolve()),
            "test": str((image_root / "test").resolve()),
            "nc": len(self.config.class_names),
            "names": list(self.config.class_names),
        }
        utils.ensure_dir(yaml_path.parent)
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        """Create a lightweight link if possible, otherwise fall back to a file copy."""
        try:
            dst.symlink_to(src)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
