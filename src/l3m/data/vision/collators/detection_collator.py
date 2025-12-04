# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch

__all__ = ["DetectionCollator"]


class DetectionCollator:
    """Pads variable-length detection targets and stacks images."""

    def __init__(self, pad_to: int | None = None, pad_label: int = -1):
        self.pad_to = pad_to
        self.pad_label = pad_label

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        boxes = [b["target_boxes"] for b in batch]
        labels = [b["target_labels"] for b in batch]
        sizes = [b.get("target_sizes") for b in batch]

        max_boxes = self.pad_to or max(box.shape[0] for box in boxes)
        batch_size = len(batch)

        padded_boxes = torch.zeros(batch_size, max_boxes, 4, dtype=boxes[0].dtype)
        padded_labels = torch.full((batch_size, max_boxes), self.pad_label, dtype=torch.long)
        boxes_mask = torch.zeros(batch_size, max_boxes, dtype=torch.bool)

        for idx, (bboxes, lbls) in enumerate(zip(boxes, labels, strict=False)):
            num = min(bboxes.shape[0], max_boxes)
            padded_boxes[idx, :num] = bboxes[:num]
            padded_labels[idx, :num] = lbls[:num]
            boxes_mask[idx, :num] = True

        images = torch.stack(images, dim=0)

        batch_dict = {
            "image": images,
            "target_boxes": padded_boxes,
            "target_labels": padded_labels,
            "target_boxes_mask": boxes_mask,
        }
        if sizes[0] is not None:
            batch_dict["target_sizes"] = torch.stack([s for s in sizes], dim=0)
        if "dataset_name" in batch[0]:
            batch_dict["dataset_name"] = batch[0]["dataset_name"]

        return batch_dict
