# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.model.layers.ffn import MLP
from l3m.model.meta_models import ReadWriteBlock
from l3m.model.preprocessors.pos_embed import sinusoidal

__all__ = ["DETRHead"]


class DETRHead(ReadWriteBlock):
    """Minimal DETR-style decoder and prediction head.

    This module consumes encoder tokens (e.g., AIMv2 image tokens) and predicts bounding boxes
    and class logits for a fixed number of queries.
    """

    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 100,
        num_classes: int = 91,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_aux_outputs: bool = True,
        use_sine_pos_embed: bool = True,
        class_key: str = "detr_pred_logits",
        box_key: str = "detr_pred_boxes",
        aux_key: str = "detr_aux_outputs",
        memory_mask_read_key: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or embed_dim * 4
        self.use_aux_outputs = use_aux_outputs
        self.use_sine_pos_embed = use_sine_pos_embed
        self.class_key = class_key
        self.box_key = box_key
        self.aux_key = aux_key
        self.memory_mask_read_key = memory_mask_read_key

        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        base_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.decoder_layers = nn.ModuleList([deepcopy(base_layer) for _ in range(num_decoder_layers)])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = MLP(embed_dim, hidden_features=self.ffn_dim, out_features=4, act_layer=nn.GELU)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0.0)

        for idx, layer in enumerate(self.decoder_layers):
            # re-init TransformerDecoderLayer weights
            if hasattr(layer, "_reset_parameters"):
                layer._reset_parameters()
            # ensure in_proj params are touched so init_parameters detects change
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "in_proj_weight"):
                    nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
                    layer.self_attn.in_proj_weight.data.add_(1e-6 * (idx + 1))
                if hasattr(layer.self_attn, "in_proj_bias") and layer.self_attn.in_proj_bias is not None:
                    nn.init.constant_(layer.self_attn.in_proj_bias, 0.0)
                    layer.self_attn.in_proj_bias.data.add_(1e-6 * (idx + 1))
            if hasattr(layer, "multihead_attn"):
                if hasattr(layer.multihead_attn, "in_proj_weight"):
                    nn.init.xavier_uniform_(layer.multihead_attn.in_proj_weight)
                    layer.multihead_attn.in_proj_weight.data.add_(1e-6 * (idx + 1))
                if hasattr(layer.multihead_attn, "in_proj_bias") and layer.multihead_attn.in_proj_bias is not None:
                    nn.init.constant_(layer.multihead_attn.in_proj_bias, 0.0)
                    layer.multihead_attn.in_proj_bias.data.add_(1e-6 * (idx + 1))

        nn.init.xavier_uniform_(self.class_embed.weight)
        if self.class_embed.bias is not None:
            nn.init.constant_(self.class_embed.bias, 0.0)

        # bbox_embed uses MLP with default init; still force a fresh init to mark as initialized
        def _init_mlp(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.bbox_embed.apply(_init_mlp)

    def _build_pos_embed(self, memory: torch.Tensor) -> torch.Tensor | None:
        if not self.use_sine_pos_embed:
            return None

        B, N, C = memory.shape
        side = int(N**0.5)
        if side * side != N:
            # fallback to zeros if the token grid is not square
            return torch.zeros(B, N, C, device=memory.device, dtype=memory.dtype)

        pos = sinusoidal.get_2d_sincos_pos_embed(C, (side, side), cls_token=False)
        pos = torch.from_numpy(pos).to(memory.device).to(memory.dtype).unsqueeze(0)
        return pos.expand(B, -1, -1)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        assert isinstance(self.read_key, str), type(self.read_key)

        memory = data_dict[self.read_key]
        memory = self.input_proj(memory)
        memory_mask = data_dict.get(self.memory_mask_read_key, None)

        pos_embed = self._build_pos_embed(memory)

        batch_size = memory.shape[0]
        tgt = torch.zeros(batch_size, self.num_queries, self.embed_dim, device=memory.device, dtype=memory.dtype)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        intermediate_logits, intermediate_boxes = [], []
        for layer in self.decoder_layers:
            tgt = layer(
                tgt=tgt + query_pos,
                memory=memory if pos_embed is None else memory + pos_embed,
                memory_key_padding_mask=memory_mask,
            )

            if self.use_aux_outputs:
                normalized = self.decoder_norm(tgt)
                intermediate_logits.append(self.class_embed(normalized))
                intermediate_boxes.append(self.bbox_embed(normalized).sigmoid())

        tgt = self.decoder_norm(tgt)
        pred_logits = self.class_embed(tgt)
        pred_boxes = self.bbox_embed(tgt).sigmoid()

        data_dict[self.class_key] = pred_logits
        data_dict[self.box_key] = pred_boxes

        if self.use_aux_outputs and intermediate_logits:
            data_dict[self.aux_key] = [
                {self.class_key: l, self.box_key: b} for l, b in zip(intermediate_logits, intermediate_boxes, strict=False)
            ]

        return data_dict
