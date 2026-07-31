#!/usr/bin/env python3
"""Regression tests for reusing Qwen2-VL code paths on Qwen2.5-VL."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.models import get_model
from qmllm.methods.mbq.quantize.auto_scale import auto_scale_block
from qmllm.methods.mbq.quantize.pre_quant import (
    _collect_internvl2_linear_scores,
    get_blocks,
    move_embed,
)
from qmllm.models import get_process_model


class _FakeSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x, **kwargs):
        return self.o_proj(self.v_proj(x))


class _FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x))


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = _FakeSelfAttention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = _FakeMlp()


class _FakeTextModel(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.embed_tokens = nn.Embedding(8, 4)
        self.rotary_emb = nn.Identity()


class _FakeQwen2_5InnerModel(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.language_model = _FakeTextModel(layers)


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.model = _FakeQwen2_5InnerModel(layers)


class Qwen2_5VLAdaptationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.layer = Qwen2_5_VLDecoderLayer()
        self.input_feat = {
            "self_attn.q_proj": torch.randn(1, 2, 4),
            "self_attn.k_proj": torch.randn(1, 2, 4),
            "self_attn.v_proj": torch.randn(1, 2, 4),
            "self_attn.o_proj": torch.randn(1, 2, 4),
            "mlp.gate_proj": torch.randn(1, 2, 4),
            "mlp.up_proj": torch.randn(1, 2, 4),
            "mlp.down_proj": torch.randn(1, 2, 8),
        }
        self.q_config = {
            "zero_point": True,
            "q_group_size": -1,
            "double_quant": False,
            "double_quant_config": {},
        }

    def test_qwen2_5_vl_registries_reuse_qwen2_vl_wrapper(self) -> None:
        self.assertIs(get_model("qwen2_5_vl"), get_model("qwen2_vl"))
        self.assertIs(get_process_model("qwen2_5_vl"), get_process_model("qwen2_vl"))

    def test_collects_qwen2_5_vl_linear_candidates(self) -> None:
        entries = _collect_internvl2_linear_scores(
            layer=self.layer,
            layer_name="model.language_model.layers.0",
            input_feat=self.input_feat,
            w_bit=3,
            q_config=self.q_config,
            ans_mask=torch.tensor([[False, True]]),
            vis_mask=torch.tensor([[True, False]]),
            reweight_ratio_dict={
                "attn_in": 1.1,
                "attn_out": 1.2,
                "mlp_in": 1.3,
                "mlp_out": 1.4,
            },
            reweight_group=True,
        )

        self.assertEqual(
            {item["name"] for item in entries},
            {
                "model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.self_attn.k_proj",
                "model.language_model.layers.0.self_attn.v_proj",
                "model.language_model.layers.0.self_attn.o_proj",
                "model.language_model.layers.0.mlp.gate_proj",
                "model.language_model.layers.0.mlp.up_proj",
                "model.language_model.layers.0.mlp.down_proj",
            },
        )

    def test_qwen2_5_vl_scale_search_uses_qwen_projection_layout(self) -> None:
        scales = auto_scale_block(
            self.layer,
            module_kwargs={},
            w_bit=3,
            q_config=self.q_config,
            input_feat=self.input_feat,
            ans_mask=torch.tensor([[False, True]]),
            vis_mask=torch.tensor([[True, False]]),
            reweight_ratio_dict={
                "attn_in": 1.1,
                "attn_out": 1.2,
                "mlp_in": 1.3,
                "mlp_out": 1.4,
            },
            loss_mode="mae",
            scale_search_config={"act_stat": "global"},
        )

        self.assertEqual(
            [prev_op for prev_op, _layers, _scale in scales],
            [
                "input_layernorm",
                "self_attn.v_proj",
                "post_attention_layernorm",
                "mlp.up_proj",
            ],
        )

    def test_get_blocks_and_move_embed_use_qwen2_5_language_model(self) -> None:
        layers = [self.layer]
        model = Qwen2_5_VLForConditionalGeneration(layers)

        self.assertIs(get_blocks(model), model.model.language_model.layers)

        original_embed = model.model.language_model.embed_tokens
        move_embed(model, torch.device("cpu"))
        self.assertIs(model.model.language_model.embed_tokens, original_embed)


if __name__ == "__main__":
    unittest.main()
