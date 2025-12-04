# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["HungarianMatcher", "DETRLoss"]


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x0, y0 = cx - 0.5 * w, cy - 0.5 * h
    x1, y1 = cx + 0.5 * w, cy + 0.5 * h
    return torch.stack((x0, y0, x1, y1), dim=-1)


def box_xywh_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, w, h = boxes.unbind(-1)
    cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
    return torch.stack((cx, cy, w, h), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)

    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-7)


class HungarianMatcher(nn.Module):
    """Computes an assignment between targets and predictions using the Hungarian algorithm."""

    def __init__(self, class_cost: float = 1.0, bbox_cost: float = 1.0, giou_cost: float = 1.0):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        assert "pred_logits" in outputs and "pred_boxes" in outputs
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, num_queries, num_classes+1]
        out_bbox = outputs["pred_boxes"]

        batch_indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_id in range(out_prob.shape[0]):
            tgt_ids = targets[batch_id]["labels"]
            tgt_bbox = targets[batch_id]["boxes"]

            if tgt_bbox.numel() == 0:
                empty = torch.zeros(0, dtype=torch.int64, device=out_bbox.device)
                batch_indices.append((empty, empty))
                continue

            cost_class = -out_prob[batch_id][:, tgt_ids]  # higher prob -> lower cost
            cost_bbox = torch.cdist(out_bbox[batch_id], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[batch_id]),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            cost_matrix = self.class_cost * cost_class + self.bbox_cost * cost_bbox + self.giou_cost * cost_giou
            src_ind, tgt_ind = linear_sum_assignment(cost_matrix.cpu())

            batch_indices.append(
                (
                    torch.as_tensor(src_ind, dtype=torch.int64, device=out_bbox.device),
                    torch.as_tensor(tgt_ind, dtype=torch.int64, device=out_bbox.device),
                )
            )

        return batch_indices


class DETRLoss(ReadWriteBlock):
    """Computes DETR loss (classification + L1 + GIoU) with Hungarian matching."""

    def __init__(
        self,
        matcher: HungarianMatcher,
        num_classes: int,
        eos_coef: float = 0.1,
        logits_read_key: str = "detr_pred_logits",
        boxes_read_key: str = "detr_pred_boxes",
        aux_read_key: str = "detr_aux_outputs",
        target_boxes_key: str = "target_boxes",
        target_labels_key: str = "target_labels",
        target_mask_key: str = "target_boxes_mask",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.logits_read_key = logits_read_key
        self.boxes_read_key = boxes_read_key
        self.aux_read_key = aux_read_key
        self.target_boxes_key = target_boxes_key
        self.target_labels_key = target_labels_key
        self.target_mask_key = target_mask_key

        class_weights = torch.ones(num_classes + 1)
        class_weights[-1] = eos_coef
        self.register_buffer("class_weights", class_weights)

    def forward(self, data_dict: DATA_DICT, model: nn.Module | None = None, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        outputs = {
            "pred_logits": data_dict[self.logits_read_key],
            "pred_boxes": data_dict[self.boxes_read_key],
        }
        if self.aux_read_key in data_dict:
            outputs["aux_outputs"] = data_dict[self.aux_read_key]

        targets = self._build_targets(data_dict)
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes, 1)

        losses, metrics = {}, {}
        ce_losses, metrics_ce = self.loss_labels(outputs, targets, indices)
        losses.update(ce_losses)
        losses["loss_bbox"], losses["loss_giou"], metrics_bbox = self.loss_boxes(outputs, targets, indices, num_boxes)
        metrics.update(metrics_ce)
        metrics.update(metrics_bbox)

        if "aux_outputs" in outputs:
            for layer_id, aux_output in enumerate(outputs["aux_outputs"]):
                layer_losses, layer_metrics = self._forward_aux(aux_output, targets, num_boxes, layer_id)
                losses.update(layer_losses)
                metrics.update(layer_metrics)

        total_loss = sum(losses.values())
        return total_loss, metrics

    def _forward_aux(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        num_boxes: int,
        layer_id: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        aux_outputs = {
            "pred_logits": outputs[self.logits_read_key],
            "pred_boxes": outputs[self.boxes_read_key],
        }
        indices = self.matcher(aux_outputs, targets)
        layer_losses, layer_metrics = {}, {}

        ce, metrics_ce = self.loss_labels(aux_outputs, targets, indices, postfix=f"_{layer_id}")
        bbox, giou, metrics_bbox = self.loss_boxes(aux_outputs, targets, indices, num_boxes, postfix=f"_{layer_id}")

        layer_losses.update(ce)
        layer_losses.update({"loss_bbox_aux_" + str(layer_id): bbox, "loss_giou_aux_" + str(layer_id): giou})
        layer_metrics.update(metrics_ce)
        layer_metrics.update(metrics_bbox)
        return layer_losses, layer_metrics

    def _build_targets(self, data_dict: DATA_DICT) -> list[dict[str, torch.Tensor]]:
        target_boxes = data_dict[self.target_boxes_key]
        target_labels = data_dict[self.target_labels_key]
        mask = data_dict.get(self.target_mask_key, torch.ones_like(target_labels, dtype=torch.bool))

        targets: list[dict[str, torch.Tensor]] = []
        for boxes, labels, valid_mask in zip(target_boxes, target_labels, mask, strict=False):
            keep = valid_mask.bool()
            targets.append({"boxes": boxes[keep], "labels": labels[keep]})
        return targets

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        postfix: str = "",
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        if idx[0].numel() > 0:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices, strict=False)], dim=0)
            target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.class_weights)
        metrics = {f"loss_ce{postfix}": loss_ce.item()}
        return {f"loss_ce{postfix}": loss_ce}, metrics

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
        postfix: str = "",
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            metrics = {f"loss_bbox{postfix}": 0.0, f"loss_giou{postfix}": 0.0}
            return zero, zero, metrics

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices, strict=False)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes

        loss_giou = 1.0 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = loss_giou.sum() / num_boxes

        metrics = {
            f"loss_bbox{postfix}": loss_bbox.item(),
            f"loss_giou{postfix}": loss_giou.item(),
        }
        return loss_bbox, loss_giou, metrics

    @staticmethod
    def _get_src_permutation_idx(
        indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(indices) == 0 or all(src.numel() == 0 for src, _ in indices):
            device = indices[0][0].device if len(indices) > 0 else torch.device("cpu")
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            return empty, empty

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)], dim=0)
        src_idx = torch.cat([src for (src, _) in indices], dim=0)
        return batch_idx, src_idx
