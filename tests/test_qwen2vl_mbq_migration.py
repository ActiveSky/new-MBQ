#!/usr/bin/env python3
"""Regression tests for running InternVL2 MBQ extensions on Qwen2-VL."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from qmllm.methods.mbq.quantize.auto_scale import auto_scale_block
from qmllm.methods.mbq.quantize.pre_quant import (
    _collect_internvl2_linear_scores,
    get_blocks as get_mbq_blocks,
    move_embed,
)
from qmllm.methods.rtn.quantizer import get_blocks as get_rtn_blocks
from qmllm.methods.smoothquant.quantize.quantizer import get_blocks as get_smoothquant_blocks


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


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = _FakeSelfAttention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = _FakeMlp()


class _FakeQwen2VLTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([Qwen2VLDecoderLayer()])
        self.rotary_emb = nn.Identity()


class _FakeQwen2VLCoreModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeQwen2VLTextModel()


class Qwen2VLForConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwen2VLCoreModel()


class Qwen2VLMbqMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.layer = Qwen2VLDecoderLayer()
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

    def test_collects_qwen2vl_low_rank_and_mixed_bit_candidates(self) -> None:
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

        names = {item["name"] for item in entries}
        self.assertEqual(
            names,
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
        families = {item["name"]: item["module_family"] for item in entries}
        self.assertEqual(
            families["model.language_model.layers.0.self_attn.q_proj"], "attn_in"
        )
        self.assertEqual(
            families["model.language_model.layers.0.self_attn.o_proj"], "attn_out"
        )
        self.assertEqual(
            families["model.language_model.layers.0.mlp.gate_proj"], "mlp_in"
        )
        self.assertEqual(
            families["model.language_model.layers.0.mlp.down_proj"], "mlp_out"
        )

    def test_qwen2vl_scale_search_accepts_group_reweight_ratios(self) -> None:
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

        prev_ops = [prev_op for prev_op, _layers, _scale in scales]
        self.assertEqual(
            prev_ops,
            [
                "input_layernorm",
                "self_attn.v_proj",
                "post_attention_layernorm",
                "mlp.up_proj",
            ],
        )

    def test_quantizers_use_qwen2vl_language_model_layers(self) -> None:
        model = Qwen2VLForConditionalGeneration()
        expected_layers = model.model.language_model.layers

        self.assertIs(get_mbq_blocks(model), expected_layers)
        self.assertIs(get_rtn_blocks(model), expected_layers)
        self.assertIs(get_smoothquant_blocks(model), expected_layers)

        move_embed(model, "cpu")
        self.assertEqual(model.model.language_model.embed_tokens.weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
