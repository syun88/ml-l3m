# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from __future__ import annotations

import random
from typing import Any, Callable

import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import InterpolationMode

from l3m.constants.typing import DATA_DICT

__all__ = ["HuggingFaceDetectionDataset", "LocalCocoDetectionDataset", "DetectionTransform"]


def _xywh_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, w, h = boxes.unbind(-1)
    cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
    return torch.stack((cx, cy, w, h), dim=-1)


def _maybe_get(objects: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in objects:
            return objects[key]
    raise KeyError(f"None of the keys {keys} were found in the objects dict.")


class DetectionTransform:
    """Resize, normalize, and optionally flip images while adjusting bounding boxes."""

    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        hflip_prob: float = 0.0,
    ):
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.mean = mean
        self.std = std
        self.hflip_prob = hflip_prob
        self.interpolation = InterpolationMode.BICUBIC

    def __call__(
        self,
        image,
        target: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        orig_w, orig_h = image.size
        boxes = target["boxes"]
        labels = target["labels"]

        new_h, new_w = self.image_size
        image = TF.resize(image, size=[new_h, new_w], interpolation=self.interpolation, antialias=True)

        scale = torch.tensor([new_w / orig_w, new_h / orig_h, new_w / orig_w, new_h / orig_h], dtype=torch.float32)
        boxes = boxes * scale

        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            boxes[:, 0] = new_w - boxes[:, 0] - boxes[:, 2]

        boxes = _xywh_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([new_w, new_h, new_w, new_h], dtype=torch.float32)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        target = {"boxes": boxes, "labels": labels}
        return image, target


class HuggingFaceDetectionDataset:
    """HuggingFace dataset wrapper for object detection tasks."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transforms: Callable[[Any, dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, torch.Tensor]]] | None = None,
        max_boxes: int = 100,
        image_key: str = "image",
        objects_key: str = "objects",
        bbox_key: str = "bbox",
        label_key: str = "label",
        bbox_format: str = "xywh",
        dataset_kwargs: dict[str, Any] | None = None,
    ):
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name, split=split, **(dataset_kwargs or {}))
        self.transforms = transforms
        self.max_boxes = max_boxes
        self.image_key = image_key
        self.objects_key = objects_key
        self.bbox_key = bbox_key
        self.label_key = label_key
        self.bbox_format = bbox_format

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DATA_DICT:
        sample = self.dataset[index]
        image = sample[self.image_key]
        objects = sample[self.objects_key]

        boxes = torch.tensor(_maybe_get(objects, self.bbox_key, "bbox"), dtype=torch.float32)
        labels = torch.tensor(_maybe_get(objects, self.label_key, "category", "label"), dtype=torch.int64)

        if self.bbox_format == "xyxy":
            boxes = torch.stack(
                (boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]),
                dim=-1,
            )

        # trim to max boxes early to keep memory predictable
        boxes = boxes[: self.max_boxes]
        labels = labels[: self.max_boxes]

        target = {"boxes": boxes, "labels": labels}
        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            w, h = image.size
            target["boxes"] = _xywh_to_cxcywh(boxes) / torch.tensor([w, h, w, h], dtype=torch.float32)
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        sample_dict: DATA_DICT = {
            "image": image,
            "target_boxes": target["boxes"],
            "target_labels": target["labels"],
            "target_sizes": torch.tensor([image.shape[1], image.shape[2]], dtype=torch.float32),
            "dataset_name": self.dataset_name,
        }
        return sample_dict


class LocalCocoDetectionDataset:
    """Dataset wrapper for local COCO annotations using torchvision's CocoDetection."""

    def __init__(
        self,
        image_root: str,
        ann_file: str,
        transforms: Callable[[Any, dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, torch.Tensor]]],
        max_boxes: int = 100,
        dataset_name: str = "coco",
    ):
        self.dataset = CocoDetection(root=image_root, annFile=ann_file)
        self.transforms = transforms
        self.max_boxes = max_boxes
        self.dataset_name = dataset_name
        # Build contiguous label mapping (COCO category ids are not 0..N-1)
        cat_ids = sorted(self.dataset.coco.cats.keys())
        self.catid_to_contig = {cid: idx for idx, cid in enumerate(cat_ids)}
        self.num_classes = len(self.catid_to_contig)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DATA_DICT:
        image, annotations = self.dataset[index]

        # COCO stores bboxes as [x, y, w, h]
        boxes = []
        labels = []
        for ann in annotations:
            if "bbox" in ann and "category_id" in ann:
                cid = ann["category_id"]
                if cid not in self.catid_to_contig:
                    continue
                boxes.append(ann["bbox"])
                labels.append(self.catid_to_contig[cid])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)[: self.max_boxes]
            labels = torch.tensor(labels, dtype=torch.int64)[: self.max_boxes]

        target = {"boxes": boxes, "labels": labels}
        image, target = self.transforms(image, target)

        sample_dict: DATA_DICT = {
            "image": image,
            "target_boxes": target["boxes"],
            "target_labels": target["labels"],
            "target_sizes": torch.tensor([image.shape[1], image.shape[2]], dtype=torch.float32),
            "dataset_name": self.dataset_name,
        }
        return sample_dict
