#!/usr/bin/env python3
"""Regression tests for InternVL3's InternVL-chat + Qwen2 MBQ adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer as HFQwen2DecoderLayer,
    Qwen2RotaryEmbedding,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.models import get_model
from qmllm.methods.mbq.quantize.pre_quant import (
    _extract_layer_hidden_states,
    get_blocks,
    move_embed,
)
from qmllm.models import get_process_model
from qmllm.models.internvl2.conversation import get_conv_template
from qmllm.models.internvl3 import InternVL3
from qmllm.models.internvl3.internvl3 import preprocess_internvl3


class _FakeInternVL3Config:
    template = "internvl2_5"
    force_image_size = 448
    pad2square = False
    dynamic_image_size = True
    use_thumbnail = True
    min_dynamic_patch = 1
    max_dynamic_patch = 12


class Qwen2DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class _FakeQwen2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.rotary_emb = nn.Identity()
        self.layers = nn.ModuleList([Qwen2DecoderLayer()])


class _FakeQwen2ForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwen2Model()


class InternVLChatModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeInternVL3Config()
        self.num_image_token = 4
        self.language_model = _FakeQwen2ForCausalLM()
        self.vision_model = nn.Identity()
        self.mlp1 = nn.Identity()


class _FakeTokenizer:
    pad_token_id = 7


class InternVL3AdaptationTest(unittest.TestCase):
    def test_process_registry_exposes_internvl3_adapter(self) -> None:
        self.assertIs(get_process_model("internvl3"), InternVL3)
        self.assertEqual(get_model("internvl3").__name__, "InternVL3")

    def test_wrapper_reads_internvl3_config_and_template(self) -> None:
        wrapper = InternVL3(InternVLChatModel(), _FakeTokenizer())

        self.assertEqual(wrapper.template_name, "internvl2_5")
        self.assertEqual(wrapper.image_size, 448)
        self.assertEqual(wrapper.max_dynamic_patch, 1)
        self.assertTrue(wrapper.dynamic_image_size)
        self.assertEqual(wrapper._get_data_collator_pad_id(), 7)
        self.assertEqual(get_conv_template("internvl2_5").name, "internvl2_5")
        self.assertEqual(wrapper.get_preprocess_function().__name__, "preprocess_internvl3")

    def test_qwen2_chatml_preprocessing_keeps_assistant_labels(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL3-1B-Instruct",
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        result = preprocess_internvl3(
            "internvl2_5",
            [[
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "A test image."},
            ]],
            tokenizer,
            [4],
            group_by_length=True,
        )

        self.assertEqual(result["input_ids"].shape, result["labels"].shape)
        self.assertTrue((result["labels"] != -100).any())
        image_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.assertEqual(int((result["input_ids"] == image_context_id).sum()), 4)

    def test_mbq_uses_qwen2_embeddings_inside_internvl_chat(self) -> None:
        model = InternVLChatModel()
        self.assertIs(get_blocks(model), model.language_model.model.layers)

        embedding = model.language_model.model.embed_tokens
        move_embed(model, torch.device("cpu"))
        self.assertIs(model.language_model.model.embed_tokens, embedding)
        self.assertEqual(embedding.weight.device.type, "cpu")

    def test_mbq_preserves_batch_dimension_for_tensor_decoder_outputs(self) -> None:
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        layer = HFQwen2DecoderLayer(config, layer_idx=0).eval()
        hidden_states = torch.randn(3, 7, 32)
        position_ids = torch.arange(7).unsqueeze(0).expand(3, -1)
        position_embeddings = Qwen2RotaryEmbedding(config)(
            hidden_states, position_ids
        )
        layer_output = layer(
            hidden_states, position_embeddings=position_embeddings
        )

        # Transformers 4.57's Qwen2DecoderLayer returns the tensor directly;
        # older decoder layers return it as the first tuple item.
        self.assertIsInstance(layer_output, torch.Tensor)
        self.assertIs(_extract_layer_hidden_states(layer_output), layer_output)
        self.assertEqual(
            _extract_layer_hidden_states(layer_output).shape,
            torch.Size([3, 7, 32]),
        )
        self.assertIs(
            _extract_layer_hidden_states((layer_output, None)), layer_output
        )

    def test_internvl3_eval_assets_target_internvl3(self) -> None:
        scale_config = yaml.safe_load(
            (REPO_ROOT / "configs/internvl3/MBQ_search/my_1b_weight_only_svd.yaml").read_text()
        )
        self.assertEqual(scale_config["model"], "internvl3")
        self.assertEqual(
            scale_config["model_args"], "pretrained=OpenGVLab/InternVL3-1B-Instruct"
        )
        self.assertNotIn("max_num=12", scale_config["model_args"])

        config_path = REPO_ROOT / "configs/internvl3/Eval/my_eval_ocrbench.yaml"
        config = yaml.safe_load(config_path.read_text())
        self.assertEqual(config["model"], "internvl3")
        self.assertEqual(
            config["model_args"], "pretrained=OpenGVLab/InternVL3-1B-Instruct"
        )

        script = (REPO_ROOT / "configs/internvl3/scripts/2_run_quant_eval.sh").read_text()
        self.assertIn("configs/internvl3/Eval/my_eval_ocrbench.yaml", script)
        self.assertNotIn("configs/internvl2/", script)


if __name__ == "__main__":
    unittest.main()
