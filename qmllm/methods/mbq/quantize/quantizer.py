import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from qmllm.methods.mbq.quantize.qmodule import ScaledActivation
from qmllm.utils.search import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear, WOQLowRankLinear

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
    low_rank_results=None,
    layer_bit_map=None,
    linear_bit_map=None,
):
    from .pre_quant import get_blocks, get_named_linears
    from qmllm.utils.search import get_op_name

    layers = get_blocks(model)
    low_rank_results = low_rank_results or []
    override_bit_map = {}
    override_bit_map.update(layer_bit_map or {})
    override_bit_map.update(linear_bit_map or {})
    low_rank_map = {item["name"]: item for item in low_rank_results}
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        layer_name = get_op_name(model, layers[i])
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            full_name = layer_name + "." + n
            linear_w_bit = int(override_bit_map.get(full_name, w_bit))

            if full_name in low_rank_map:
                low_rank_state = low_rank_map[full_name]
                new_linear = WOQLowRankLinear.from_float(
                    m,
                    low_rank_state,
                    weight_quant="per_group",
                    w_bit=linear_w_bit,
                    weight_group=q_config.get("q_group_size", 128),
                )
                father_module = get_module_by_name_suffix(
                    layers[i], ".".join(n.split(".")[:-1])
                )
                setattr(father_module, n.split(".")[-1], new_linear)
                del new_linear, m
            else:
                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bits=linear_w_bit, **q_config
                )


# 下面是MBQ相比于AWQ增加的
def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


@torch.no_grad()
def pseudo_quantize_model_weight_act(
    model,
    w_bit,
    a_bit,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight activation quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            new_linear = WALinear.from_float(
                m,
                weight_quant="per_channel",
                act_quant="per_token",
                w_bit=w_bit,
                a_bit=a_bit,
            )
            father_module = get_module_by_name_suffix(
                layers[i], ".".join(n.split(".")[:-1])
            )
            setattr(father_module, n.split(".")[-1], new_linear)
            del new_linear, m
            torch.cuda.empty_cache()
