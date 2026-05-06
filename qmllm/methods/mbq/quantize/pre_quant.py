import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
from collections import defaultdict
from typing import List

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.mbq.quantize.auto_scale_wa_distort import auto_scale_block_wa_distort
from qmllm.methods.mbq.quantize.auto_scale_wa import auto_scale_block_wa
from qmllm.methods.mbq.quantize.auto_scale_distort import auto_scale_block_distort
from qmllm.methods.mbq.quantize.auto_scale import auto_scale_block, apply_scale, scale_ln_fcs, scale_fc_fc
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from .quantizer import get_module_by_name_suffix


# MBQ新增多模态输入、重加权和权重-激活量化入口
__all__ = ["run_mbq"]

# ========== SmoothQuant-style smoothing helpers ==========

@torch.no_grad()
def _compute_smooth_scales(act, weight, alpha=0.5):
    """Compute SmoothQuant smoothing scales per channel.
    S_j = (max(|X_j|))^alpha / (max(|W_j|))^(1-alpha)
    """
    act = act.to(weight.device)
    act_max = act.abs().view(-1, act.shape[-1]).amax(dim=0).float()
    weight_max = weight.abs().amax(dim=0).float()
    act_max = act_max.clamp(min=1e-5)
    weight_max = weight_max.clamp(min=1e-5)
    scales = act_max.pow(alpha) / weight_max.pow(1 - alpha)
    scales = scales.clamp(min=1e-5)
    scales = scales / (scales.max() * scales.min()).sqrt()
    return scales.to(weight.dtype)


@torch.no_grad()
def _smooth_ln_fcs(ln, fcs, input_feat, fc_names, alpha=0.5):
    """Apply SmoothQuant smoothing for a group: LN -> [FC1, FC2, ...].
    SmoothQuant formula: Y = (X/S) * (W*S)
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    if not isinstance(fc_names, list):
        fc_names = [fc_names]
    first_fc = fcs[0]
    first_name = fc_names[0]
    act = input_feat[first_name]
    weight = first_fc.weight.data
    S = _compute_smooth_scales(act, weight, alpha=alpha)
    scale_ln_fcs(ln, fcs, S)
    S = S.to(input_feat[first_name].device)
    for name in fc_names:
        if name in input_feat:
            input_feat[name] = input_feat[name].div(S.view(1, -1))


@torch.no_grad()
def _smooth_fc_fc(fc1, fc2, input_feat, fc2_name, alpha=0.5):
    """Apply SmoothQuant smoothing for a FC pair: FC1 -> FC2.
    FC1 weight[out] /= S, FC2 weight *= S
    """
    act = input_feat[fc2_name]
    weight = fc2.weight.data
    S = _compute_smooth_scales(act, weight, alpha=alpha)
    scale_fc_fc(fc1, fc2, S)
    S = S.to(input_feat[fc2_name].device)
    if fc2_name in input_feat:
        input_feat[fc2_name] = input_feat[fc2_name].div(S.view(1, -1))


@torch.no_grad()
def _apply_smooth_layer(layer, named_linears, input_feat, smooth_alpha=0.5):
    """Apply SmoothQuant-style smoothing to all Linear groups in one layer.
    Supports InternLM2DecoderLayer (InternVL2), Qwen2VLDecoderLayer, LlamaDecoderLayer.
    """

    if layer.__class__.__name__ == "InternLM2DecoderLayer":
        print(f"Applying SmoothQuant to InternLM2DecoderLayer...", flush=True)
        # InternVL2: attention_norm -> wqkv (fused QKV)
        if "attention.wqkv" in named_linears and "attention.wqkv" in input_feat:
            _smooth_ln_fcs(layer.attention_norm, [named_linears["attention.wqkv"]],
                          input_feat, ["attention.wqkv"], alpha=smooth_alpha)
        # InternVL2: wqkv -> wo
        if ("attention.wo" in named_linears and "attention.wo" in input_feat
                and "attention.wqkv" in named_linears):
            _smooth_fc_fc(named_linears["attention.wqkv"], named_linears["attention.wo"],
                         input_feat, "attention.wo", alpha=smooth_alpha)
        # InternVL2: ffn_norm -> w1, w3
        gateup_linears = []
        gateup_names = []
        for n in ["feed_forward.w1", "feed_forward.w3"]:
            if n in named_linears and n in input_feat:
                gateup_linears.append(named_linears[n])
                gateup_names.append(n)
        if gateup_linears:
            _smooth_ln_fcs(layer.ffn_norm, gateup_linears, input_feat, gateup_names, alpha=smooth_alpha)
        # InternVL2: w3 -> w2
        if ("feed_forward.w2" in named_linears and "feed_forward.w2" in input_feat
                and "feed_forward.w3" in named_linears):
            _smooth_fc_fc(named_linears["feed_forward.w3"], named_linears["feed_forward.w2"],
                         input_feat, "feed_forward.w2", alpha=smooth_alpha)

    elif layer.__class__.__name__ == "Qwen2VLDecoderLayer":
        # Qwen2VL: same structure as Llama
        qkv_linears = []
        qkv_names = []
        for n in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            if n in named_linears and n in input_feat:
                qkv_linears.append(named_linears[n])
                qkv_names.append(n)
        if qkv_linears:
            _smooth_ln_fcs(layer.input_layernorm, qkv_linears, input_feat, qkv_names, alpha=smooth_alpha)
        if ("self_attn.o_proj" in named_linears and "self_attn.o_proj" in input_feat
                and hasattr(layer.self_attn, 'v_proj')):
            _smooth_fc_fc(named_linears["self_attn.v_proj"], named_linears["self_attn.o_proj"],
                         input_feat, "self_attn.o_proj", alpha=smooth_alpha)
        gateup_linears = []
        gateup_names = []
        for n in ["mlp.gate_proj", "mlp.up_proj"]:
            if n in named_linears and n in input_feat:
                gateup_linears.append(named_linears[n])
                gateup_names.append(n)
        if gateup_linears:
            _smooth_ln_fcs(layer.post_attention_layernorm, gateup_linears, input_feat, gateup_names, alpha=smooth_alpha)
        if ("mlp.down_proj" in named_linears and "mlp.down_proj" in input_feat
                and "mlp.up_proj" in named_linears):
            _smooth_fc_fc(named_linears["mlp.up_proj"], named_linears["mlp.down_proj"],
                         input_feat, "mlp.down_proj", alpha=smooth_alpha)

    elif isinstance(layer, nn.Module) and hasattr(layer, 'input_layernorm'):
        # Generic Llama-like (LlamaDecoderLayer, etc.)
        qkv_linears = []
        qkv_names = []
        for n in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            if n in named_linears and n in input_feat:
                qkv_linears.append(named_linears[n])
                qkv_names.append(n)
        if qkv_linears:
            _smooth_ln_fcs(layer.input_layernorm, qkv_linears, input_feat, qkv_names, alpha=smooth_alpha)
        if ("self_attn.o_proj" in named_linears and "self_attn.o_proj" in input_feat
                and hasattr(layer.self_attn, 'v_proj')):
            _smooth_fc_fc(named_linears["self_attn.v_proj"], named_linears["self_attn.o_proj"],
                         input_feat, "self_attn.o_proj", alpha=smooth_alpha)
        gateup_linears = []
        gateup_names = []
        for n in ["mlp.gate_proj", "mlp.up_proj"]:
            if n in named_linears and n in input_feat:
                gateup_linears.append(named_linears[n])
                gateup_names.append(n)
        if gateup_linears:
            _smooth_ln_fcs(layer.post_attention_layernorm, gateup_linears, input_feat, gateup_names, alpha=smooth_alpha)
        if ("mlp.down_proj" in named_linears and "mlp.down_proj" in input_feat
                and "mlp.up_proj" in named_linears):
            _smooth_fc_fc(named_linears["mlp.up_proj"], named_linears["mlp.down_proj"],
                         input_feat, "mlp.down_proj", alpha=smooth_alpha)

    else:
        print(f"[SmoothQuant] Warning: unsupported layer type {type(layer).__name__}, skipping smoothing for this layer.")

# MBQ新增的
class GradCacheHook:
    def __init__(self, vis_masks, cap_masks):
        if vis_masks is None or cap_masks is None:
            raise ValueError
        self.hooks = []
        self.vis_masks = vis_masks.cpu()
        self.cap_masks = cap_masks.cpu()
        self.steps = {}
        self.grad_dict = {}


    def cache_grad_hook(self, module, inp, out, name):
        # initialize step counter, we use step counter to find the right mask for the grad
        if name not in self.steps:
            self.steps[name] = 0

        if name not in self.grad_dict:
            self.grad_dict[name] = {"vis_grad": [], "cap_grad": []}

        output_grad = out[0].float()
        step = self.steps[name]

        B, N, C = output_grad.shape

        for batch_idx in range(B):
            vis_mask = self.vis_masks[step]
            cap_mask = self.cap_masks[step]

            vis_grad = output_grad[batch_idx][vis_mask]
            cap_grad = output_grad[batch_idx][cap_mask]

            vis_grad_avg = vis_grad.abs().mean()
            cap_grad_avg = cap_grad.abs().mean()

            self.grad_dict[name]["vis_grad"].append(vis_grad_avg.detach().cpu())
            self.grad_dict[name]["cap_grad"].append(cap_grad_avg.detach().cpu())

            step = step + 1

        self.steps[name] = step


    def register_hooks(self, layers):
        for n, m in layers.named_modules():
            if isinstance(m, nn.Linear) and any([_ in n for _ in ["wo", "w2", "down_proj", "o_proj", "v_proj", "gate_proj", "up_proj", "w1", "w3"]]):
                # print(f"Registering hook for layer.{n}")
                self.hooks.append(
                    m.register_full_backward_hook(
                        functools.partial(self.cache_grad_hook, name=f"layers.{n}")
                    )
                )


    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


    def get_grad_dict(self):
        return self.grad_dict
    

    def get_avg_grad_dict(self):
        avg_grad_dict = {}

        for name, grad_values in self.grad_dict.items():
            mean_vis = torch.mean(torch.stack(grad_values["vis_grad"]))
            mean_cap = torch.mean(torch.stack(grad_values["cap_grad"]))

            avg_grad_dict[name] = {
                "vis_avg_grad": mean_vis.item(),
                "cap_avg_grad": mean_cap.item()
            }

        return avg_grad_dict
    

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

# MBQ:增加一些qwen2vl,llava的处理模块
def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVLChatModel":
        layers = model.language_model.model.layers
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers

# MBQ:增加一些qwen2vl,llava的处理模块
def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        # model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        model.model.tok_embeddings = model.model.tok_embeddings.to(device)
    elif model.__class__.__name__ == "InternVLChatModel":
        model.language_model.model.tok_embeddings = model.language_model.model.tok_embeddings.to(device)  
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif model.__class__.__name__ == "LlavaLlamaModel":
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))

# MBQ：完全新增
def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    
    return inputs, vision_mask, caption_mask


@torch.no_grad()
def run_mbq(
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit, #新增
    q_config,
    auto_scale=True,
    # 以下参数都新增
    loss_mode="mae",
    wa_quant=False,
    reweight=False,
    distort=False,
    smooth=False,
    smooth_alpha=0.5,
):
    if "bigcode" in str(model.model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model.model)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model.model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    # =======MBQ：新增
    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)

    model.to_cuda()
    try:
        model(**inputs)
    except ValueError: # work with early exit
        pass

    model.to_cpu()
    # =========新增结束
    layers[0] = layers[0].module  # restore
    inps = inps[0]
    layer_kwargs["use_cache"] = False

    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    mbq_results = {
        "scale": [],
    }

    # ===========MBQ:下面reweight和distort都是新增
    if reweight:
        model.to_cuda()
        print("Save gradient...")
        # save gradient
        grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask)        
        grad_cache.register_hooks(layers=layers)
        
        with torch.enable_grad():
            mini_batch = 1
            total_samples = next(iter(prompt_inputs.values())).shape[0]
            accum_steps = int(total_samples/mini_batch)
            
            for i in tqdm.tqdm(range(0, total_samples, mini_batch), desc="Running gradient calculation..."):
                mini_inputs = {}
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        mini_inputs[k] = inputs[k][i:i+mini_batch]
                
                outputs = model(**mini_inputs)

                loss = outputs[0]

                loss = loss / accum_steps
                loss.backward()

        model.to_cpu()
        grad_avg_dict = grad_cache.get_avg_grad_dict()
        grad_cache.remove_hooks()
        del grad_cache

        attn_list = []
        mlp_list = []

        for key_name in grad_avg_dict:
            if "down_" in key_name or "w2" in key_name:
                mlp_list.append(grad_avg_dict[key_name]["vis_avg_grad"] / grad_avg_dict[key_name]["cap_avg_grad"])
            if "o_proj" in key_name or "wo" in key_name:
                attn_list.append(grad_avg_dict[key_name]["vis_avg_grad"] / grad_avg_dict[key_name]["cap_avg_grad"])

        attn_median = np.median(attn_list)
        mlp_median = np.median(mlp_list)


    if distort:
        # assert wa_quant, "We only support distort input in weight-activation quantization!!!"
        print("Use distort input...")
        inps_distort = copy.deepcopy(inps)

    gc.collect()
    torch.cuda.empty_cache()
    # ============新增结束
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running MBQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        # =====MBQ:新增
        for k in layer_kwargs:
            if isinstance(layer_kwargs[k], torch.Tensor):
                layer_kwargs[k] = layer_kwargs[k].to(next(layer.parameters()).device)
        # ========新增结束
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        # SmoothQuant-style smoothing before scale search
        if smooth:
            print(f"Applying SmoothQuant-style smoothing to layer {i}...", flush=True)
            _apply_smooth_layer(layer, named_linears, input_feat, smooth_alpha=smooth_alpha)

        # =======新增
        if reweight:
            scale_reweight_ratio_dict = {}
            for key, value in grad_avg_dict.items():
                item_list = key.split(".")
                if str(i) in item_list:
                    if "wo" in item_list or "o_proj" in item_list:
                        scale_reweight_ratio_dict["attn"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), attn_median)
                    elif "w2" in item_list or "down_proj" in item_list:
                        scale_reweight_ratio_dict["mlp"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), mlp_median)
        else:
            scale_reweight_ratio_dict = {
                "attn": None,
                "mlp": None
            }
        # =======新增结束
        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            # ===========新增
            if not reweight:
                ans_mask = None
                vis_mask = None
            else:
                ans_mask = caption_mask
                vis_mask = vision_mask
            
            if wa_quant:
                if distort:
                    scales_list = auto_scale_block_wa_distort(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode
                    )
                else:
                    scales_list = auto_scale_block_wa(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode
                    )
            else:
                if distort:
                    scales_list = auto_scale_block_distort(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode
                    )
                # =========新增结束
                else:
                    scales_list = auto_scale_block(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        # 新增参数
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode
                    )

            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # =========新增
            if distort:
                # get distort output as next layer's input
                if wa_quant:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                        father_module = get_module_by_name_suffix(layer_q, '.'.join(n.split(".")[:-1]))
                        setattr(father_module, n.split('.')[-1], new_linear)
                        del new_linear, m
                        torch.cuda.empty_cache()
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    inps_distort = layer_q(inps_distort, **layer_kwargs)[0]
                    del layer_q 
                else:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bits=w_bit, **q_config)
                        torch.cuda.empty_cache()
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    inps_distort = layer_q(inps_distort, **layer_kwargs)[0]
                    del layer_q 
            # ===========新增结束
            # append prefix to make names global
            mbq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model.model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return mbq_results


def apply_mbq(model, mbq_results):
    apply_scale(model, mbq_results["scale"])