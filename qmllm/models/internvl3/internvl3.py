"""Calibration adapter for InternVL3's InternVL-chat + Qwen2 architecture."""

from __future__ import annotations

from typing import Dict, List

import torch

from qmllm.models.internvl2.constants import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
)
from qmllm.models.internvl2.conversation import get_conv_template
from qmllm.models.internvl2.internvl2 import InternVL2
from qmllm.utils.registry import MODEL_REGISTRY


IGNORE_TOKEN_ID = -100


def _tokenize_without_implicit_special_tokens(tokenizer, text: str) -> List[int]:
    """Tokenize an already-formatted InternVL ChatML segment exactly once."""
    return tokenizer(text, add_special_tokens=False).input_ids


def preprocess_internvl3(
    template_name,
    sources,
    tokenizer,
    num_image_token_list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict[str, torch.Tensor]:
    """Build InternVL3/Qwen2 calibration labels without length heuristics.

    InternVL2's legacy preprocessing subtracts an assumed BOS token from
    segment lengths. Qwen2 has no BOS token and uses ChatML special tokens,
    which can shift the inferred answer span by one token and mask every
    answer label. Here prompts and labels are built token-by-token: system and
    user segments are ignored, while assistant content plus its terminator is
    supervised.
    """
    conv = get_conv_template(template_name)
    role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_length = int(getattr(tokenizer, "model_max_length", 32768))

    all_input_ids: List[List[int]] = []
    all_labels: List[List[int]] = []

    for source in sources:
        if not source:
            continue
        if source[0].get("from") != "human":
            source = source[1:]

        input_ids: List[int] = []
        labels: List[int] = []

        system_segment = conv.system_template.format(
            system_message=conv.system_message
        ) + conv.sep
        system_ids = _tokenize_without_implicit_special_tokens(
            tokenizer, system_segment
        )
        input_ids.extend(system_ids)
        labels.extend([IGNORE_TOKEN_ID] * len(system_ids))

        image_index = 0
        for turn_index, sentence in enumerate(source):
            source_role = sentence.get("from")
            if source_role not in role_map:
                raise ValueError(
                    f"Unsupported InternVL3 conversation role: {source_role!r}"
                )
            role = role_map[source_role]
            if role != conv.roles[turn_index % 2]:
                raise ValueError(
                    "InternVL3 conversations must alternate human and gpt turns."
                )

            value = str(sentence.get("value", "")).strip()
            if not text_only:
                while "<image>" in value:
                    if image_index >= len(num_image_token_list):
                        raise ValueError(
                            "Conversation contains more <image> placeholders than "
                            "provided image-token counts."
                        )
                    image_tokens = (
                        f"{IMG_START_TOKEN}"
                        f"{IMG_CONTEXT_TOKEN * int(num_image_token_list[image_index])}"
                        f"{IMG_END_TOKEN}"
                    )
                    value = value.replace("<image>", image_tokens, 1)
                    image_index += 1

            prefix_ids = _tokenize_without_implicit_special_tokens(tokenizer, role)
            content_ids = _tokenize_without_implicit_special_tokens(
                tokenizer, value + conv.sep
            )
            input_ids.extend(prefix_ids)
            input_ids.extend(content_ids)

            if source_role == "gpt":
                labels.extend([IGNORE_TOKEN_ID] * len(prefix_ids))
                labels.extend(content_ids)
            else:
                labels.extend([IGNORE_TOKEN_ID] * (len(prefix_ids) + len(content_ids)))

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    if not all_input_ids:
        raise ValueError("InternVL3 preprocessing received no usable conversations.")

    # ``preprocess_data`` calls this with one sample, but padding here keeps the
    # function correct for callers that batch multiple source conversations.
    max_seq_len = max(len(item) for item in all_input_ids)
    input_tensor = torch.full(
        (len(all_input_ids), max_seq_len), pad_id, dtype=torch.long
    )
    label_tensor = torch.full(
        (len(all_labels), max_seq_len), IGNORE_TOKEN_ID, dtype=torch.long
    )
    attention_mask = torch.zeros(
        (len(all_input_ids), max_seq_len), dtype=torch.bool
    )
    for index, (input_ids, labels) in enumerate(zip(all_input_ids, all_labels)):
        length = len(input_ids)
        input_tensor[index, :length] = torch.tensor(input_ids, dtype=torch.long)
        label_tensor[index, :length] = torch.tensor(labels, dtype=torch.long)
        attention_mask[index, :length] = True

    return {
        "input_ids": input_tensor,
        "labels": label_tensor,
        "attention_mask": attention_mask,
    }


@MODEL_REGISTRY.register("internvl3")
class InternVL3(InternVL2):
    """Reuse InternVL image-token injection while reading InternVL3 settings.

    The HF model remains an ``InternVLChatModel`` and exposes the same
    ``vision_model``, ``mlp1``, ``extract_feature`` and ``language_model``
    interfaces as InternVL2.  Its language model is Qwen2 rather than
    InternLM2, so prompt metadata and embedding placement must come from the
    actual model configuration instead of the InternVL2 defaults.

    Calibration is deliberately capped at one image tile per sample.  The MBQ
    pipeline collates all calibration samples before calling ``extract_feature``;
    blindly inheriting InternVL3's evaluation-time ``max_dynamic_patch=12``
    turns a 128-sample calibration batch into more than one thousand vision
    tiles and creates an avoidable activation-memory spike.  This matches the
    established InternVL2 calibration convention.  Full multi-tile processing
    remains enabled in the lmms-eval evaluation wrapper via ``max_num=12``.
    """

    def __init__(self, model, tokenizer, processor=None):
        super().__init__(model, tokenizer, processor)

        config = getattr(model, "config", None)
        self.template_name = str(
            getattr(config, "template", None) or "internvl2_5"
        )
        self.image_size = int(
            getattr(config, "force_image_size", None)
            or getattr(config, "image_size", None)
            or 448
        )
        self.pad2square = bool(getattr(config, "pad2square", False))
        self.dynamic_image_size = bool(
            getattr(config, "dynamic_image_size", True)
        )
        self.use_thumbnail = bool(getattr(config, "use_thumbnail", True))
        self.min_dynamic_patch = int(
            getattr(config, "min_dynamic_patch", 1) or 1
        )
        self.max_dynamic_patch = 1

    def get_preprocess_function(self):
        return preprocess_internvl3

    def _get_data_collator_pad_id(self):
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
