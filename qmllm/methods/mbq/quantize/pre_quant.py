import os
import time

import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
from collections import defaultdict
from typing import List, Optional

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name, get_op_by_name

from qmllm.methods.mbq.quantize.auto_scale_wa_distort import auto_scale_block_wa_distort
from qmllm.methods.mbq.quantize.auto_scale_wa import auto_scale_block_wa
from qmllm.methods.mbq.quantize.auto_scale_distort import auto_scale_block_distort
from qmllm.methods.mbq.quantize.auto_scale import auto_scale_block, apply_scale
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from .quantizer import get_module_by_name_suffix

# MBQ新增多模态输入、重加权和权重-激活量化入口
__all__ = ["run_mbq"]


@torch.no_grad()
def _get_internvl2_candidate_specs():
    return [
        ("attention.wqkv", "attention_wqkv", "attn"),
        ("attention.wo", "attention_wo", "attn"),
        ("feed_forward.w1", "mlp_w1", "mlp"),
        ("feed_forward.w3", "mlp_w3", "mlp"),
        ("feed_forward.w2", "mlp_w2", "mlp"),
    ]


@torch.no_grad()
def _compute_multimodal_activation_aware_score(
    module_input,
    weight_fp,
    weight_q,
    token_ans_mask=None,
    token_vis_mask=None,
    reweight_ratio=None,
    timing=None,
):
    def _relative_error(flat_input, weight_fp, residual_weight, token_mask=None):
        if token_mask is not None:
            token_mask = (
                token_mask.reshape(-1).to(dtype=torch.bool).to(device=flat_input.device)
            )
            if token_mask.numel() != flat_input.shape[0]:
                print(
                    "Error: token_mask length {} does not match number of tokens {}.".format(
                        token_mask.numel(), flat_input.shape[0]
                    )
                )
                return None
            flat_input = flat_input[token_mask]
        if flat_input.numel() == 0:
            return None
        ref = torch.nn.functional.linear(flat_input, weight_fp)
        err = torch.nn.functional.linear(flat_input, residual_weight)
        numerator = err.float().pow(2).sum()
        denominator = ref.float().pow(2).sum()
        return float((numerator / denominator.clamp(min=1e-6)).item())

    flat_input = module_input.reshape(-1, module_input.shape[-1]).float()
    transfer_start = time.perf_counter()
    flat_input = flat_input.to(weight_fp.device)
    if timing is not None:
        timing["input_transfer"] += time.perf_counter() - transfer_start

    residual_weight = (weight_fp - weight_q).float()

    if token_ans_mask is None or token_vis_mask is None:
        global_start = time.perf_counter()
        global_score = _relative_error(flat_input, weight_fp, residual_weight)
        if timing is not None:
            timing["score_global"] += time.perf_counter() - global_start
        return global_score

    ans_start = time.perf_counter()
    ans_score = _relative_error(
        flat_input,
        weight_fp,
        residual_weight,
        token_mask=token_ans_mask,
    )
    if timing is not None:
        timing["score_ans"] += time.perf_counter() - ans_start

    vis_start = time.perf_counter()
    vis_score = _relative_error(
        flat_input,
        weight_fp,
        residual_weight,
        token_mask=token_vis_mask,
    )
    if timing is not None:
        timing["score_vis"] += time.perf_counter() - vis_start

    if ans_score is None and vis_score is None:
        return None
    if ans_score is None:
        ans_score = 0.0
    if vis_score is None:
        vis_score = 0.0

    if reweight_ratio is None:
        return ans_score + vis_score
    return ans_score + float(reweight_ratio) * vis_score


@torch.no_grad()
def _collect_internvl2_linear_scores(
    layer,
    layer_name,
    input_feat,
    w_bit,
    q_config,
    ans_mask=None,
    vis_mask=None,
    reweight_ratio_dict=None,
    emit_timing=False,
):
    candidates = []
    if layer.__class__.__name__ != "InternLM2DecoderLayer":
        return candidates

    timing = {
        "pseudo_quant": 0.0,
        "input_transfer": 0.0,
        "score_global": 0.0,
        "score_ans": 0.0,
        "score_vis": 0.0,
    }
    call_start = time.perf_counter()

    for module_name, module_type, module_family in _get_internvl2_candidate_specs():
        if module_name not in input_feat:
            continue

        module = get_op_by_name(layer, module_name)
        if module is None or not isinstance(module, nn.Linear):
            continue

        weight_fp = module.weight.data.detach().float()
        quant_start = time.perf_counter()
        weight_q = pseudo_quantize_tensor(
            weight_fp, n_bits=w_bit, inplace=False, **q_config
        )
        timing["pseudo_quant"] += time.perf_counter() - quant_start

        reweight_ratio = None
        if reweight_ratio_dict is not None:
            reweight_ratio = reweight_ratio_dict.get(module_family)

        score = _compute_multimodal_activation_aware_score(
            input_feat[module_name],
            weight_fp,
            weight_q,
            token_ans_mask=ans_mask,
            token_vis_mask=vis_mask,
            reweight_ratio=reweight_ratio,
            timing=timing,
        )
        if score is None:
            continue

        candidates.append(
            {
                "name": layer_name + "." + module_name,
                "score": float(score),
                "module_type": module_type,
                "module_family": module_family,
                "w_bit": int(w_bit),
            }
        )

    if emit_timing:
        call_elapsed = time.perf_counter() - call_start
        print(
            "[Timing] layer {} linear scores: total={:.3f}s, "
            "pseudo_quant={:.3f}s, input_transfer={:.3f}s, "
            "score_global={:.3f}s, score_ans={:.3f}s, score_vis={:.3f}s".format(
                layer_name,
                call_elapsed,
                timing["pseudo_quant"],
                timing["input_transfer"],
                timing["score_global"],
                timing["score_ans"],
                timing["score_vis"],
            ),
            flush=True,
        )

    return candidates


def _build_linear_bit_map(
    linear_score_entries,
    default_w_bit,
    high_w_bit,
    keep_ratio,
):
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    sorted_entries = sorted(
        linear_score_entries, key=lambda item: item["score"], reverse=True
    )
    keep_count = min(
        len(sorted_entries),
        max(0, int(np.ceil(len(sorted_entries) * keep_ratio))),
    )
    high_precision_names = {item["name"] for item in sorted_entries[:keep_count]}
    linear_bit_map = {}
    for item in sorted_entries:
        linear_bit_map[item["name"]] = (
            int(high_w_bit)
            if item["name"] in high_precision_names
            else int(default_w_bit)
        )
    return linear_bit_map


@torch.no_grad()
def _build_low_rank_states(
    model,
    candidates,
    w_bit,
    q_config,
    attn_topk_ratio=0.4,
    mlp_topk_ratio=0.4,
):
    """根据候选层分数，真正构建可保存的 low-rank SVD 状态。

    这个函数会先按 score 选出 top-k 候选层，再重新从模型里取回对应模块，
    计算 `W - W_q` 的残差矩阵，并对残差做截断 SVD：

        residual ≈ up @ down

    最终得到的 `up` 和 `down` 会被保存到 `mbq_results["low_rank"]` 中，
    供后续加载模型时构造 `WOQLowRankLinear` 使用。

    参数:
        model: 当前已经 apply_scale 后的完整语言模型。
        candidates: 所有候选层的轻量信息列表，每项至少包含 name / score / rank。
        attn_topk_ratio: attention 族候选层 top-k 比例。
        mlp_topk_ratio: MLP 族候选层 top-k 比例。
        w_bit: 权重量化 bit 数。
        q_config: 当前量化配置，如 group size、zero point 等。

    返回:
        low_rank_results: 每个选中层对应一个字典，包含 name / rank / score / up / down。
    """

    def _select_low_rank_candidates(
        candidates,
        attn_topk_ratio=0.4,
        mlp_topk_ratio=0.4,
    ):
        def _take_topk(group_candidates, ratio):
            ratio = max(0.0, min(1.0, float(ratio)))
            if not group_candidates or ratio <= 0:
                return []
            topk_count = int(np.ceil(len(group_candidates) * ratio))
            topk_count = min(len(group_candidates), max(0, topk_count))
            return sorted(
                group_candidates, key=lambda item: item["score"], reverse=True
            )[:topk_count]

        attn_ratio = float(attn_topk_ratio)
        mlp_ratio = float(mlp_topk_ratio)
        attn_candidates = [
            item for item in candidates if item.get("module_family") == "attn"
        ]
        mlp_candidates = [
            item for item in candidates if item.get("module_family") == "mlp"
        ]

        selected = []
        selected.extend(_take_topk(attn_candidates, attn_ratio))
        selected.extend(_take_topk(mlp_candidates, mlp_ratio))
        return sorted(selected, key=lambda item: item["score"], reverse=True)

    # 如果没有任何候选层，直接返回空列表。
    if not candidates:
        return []

    candidate_type_counter = defaultdict(int)
    for item in candidates:
        candidate_type_counter[item.get("module_type", "unknown")] += 1
    print(
        "Low-rank candidates collected: total={}, detail={}".format(
            len(candidates), dict(candidate_type_counter)
        )
    )

    selected = _select_low_rank_candidates(
        candidates,
        attn_topk_ratio=attn_topk_ratio,
        mlp_topk_ratio=mlp_topk_ratio,
    )
    if not selected:
        return []

    selected_type_counter = defaultdict(int)
    for item in selected:
        selected_type_counter[item.get("module_type", "unknown")] += 1
    print(
        "Low-rank candidates selected: total={}, detail={}".format(
            len(selected), dict(selected_type_counter)
        )
    )
    # 初始化最终的 low-rank 状态列表。
    low_rank_results = []
    # 逐个为选中的层构建 low-rank 补偿矩阵。
    for item in tqdm.tqdm(selected, desc="Building low-rank states"):
        # 根据完整模块名，从模型里重新取回该线性层。
        module = get_op_by_name(model, item["name"])
        # 读取 apply_scale 后的浮点权重，作为 low-rank 补偿的目标基准。
        weight_fp = module.weight.data.detach().float()
        # 用与真实部署一致的量化配置重新计算量化主权重。
        weight_q = pseudo_quantize_tensor(
            weight_fp, n_bits=w_bit, inplace=False, **q_config
        )
        # 构造要被 low-rank 支路近似的残差矩阵。
        residual = (weight_fp - weight_q).float()
        # 理论上 rank 不能超过矩阵的最小维度，因此这里先求出允许的最大 rank。
        max_rank = min(residual.shape[0], residual.shape[1])
        # 当前层实际使用的 rank，取“用户给的固定 rank”和“矩阵允许的最大 rank”的较小值。
        rank = min(int(item["rank"]), max_rank)
        # 如果 rank 非法或为 0，直接跳过该层。
        if rank <= 0:
            continue

        # 对残差矩阵做低秩分解：优先在 GPU 上做截断 SVD，只在必要时回退到完整 SVD。
        svd_device = (
            torch.device("cuda") if torch.cuda.is_available() else residual.device
        )
        residual_svd = residual.to(svd_device, non_blocking=True)
        try:
            if rank < max_rank:
                # q 作为随机 SVD 的探测维度，略大于目标 rank 以提升近似质量。
                svd_q = min(max(rank + 8, int(rank * 1.5)), max_rank - 1)
                svd_q = max(rank, svd_q)
                U, S, V = torch.svd_lowrank(residual_svd, q=svd_q, niter=2)
                # torch.svd_lowrank 不保证按奇异值大小排序，这里先排序再截断。
                order = torch.argsort(S, descending=True)
                U = U[:, order]
                S = S[order]
                Vh = V[:, order].transpose(0, 1)
            else:
                U, S, Vh = torch.linalg.svd(residual_svd, full_matrices=False)
        except RuntimeError:
            # 如果随机 SVD 在当前设备上失败，就退回到标准 SVD 保证结果可用。
            print(
                "Randomized SVD failed for layer {}, falling back to full SVD.".format(
                    item["name"]
                )
            )
            U, S, Vh = torch.linalg.svd(residual_svd, full_matrices=False)
        finally:
            # 仅释放引用；不在循环内做 empty_cache，避免每层都触发昂贵的 GPU 内存整理。
            if residual_svd.device.type == "cuda":
                del residual_svd

        # 只保留前 rank 个最重要的奇异向量。
        U_r = U[:, :rank]
        # 只保留前 rank 个奇异值。
        S_r = S[:rank]
        # 只保留前 rank 个右奇异向量。
        Vh_r = Vh[:rank, :]
        # 将奇异值开平方，便于把 diag(S) 均分到 up 和 down 两边。
        sqrt_s = torch.sqrt(S_r)
        # 构造上投影矩阵 up，使得 up.shape = [out_features, rank]。
        up = U_r * sqrt_s.unsqueeze(0)
        # 构造下投影矩阵 down，使得 down.shape = [rank, in_features]。
        down = sqrt_s.unsqueeze(1) * Vh_r

        # 保存该层的 low-rank 状态；后续加载量化模型时会用这些张量替换原始线性层。
        low_rank_results.append(
            {
                # 保存完整模块名，便于加载时精确匹配目标层。
                "name": item["name"],
                # 保存最终实际使用的 rank。
                "rank": rank,
                # 把候选打分一并保存，方便后续分析和调试。
                "score": item["score"],
                # 保存模块类型，便于后续分析 attention / MLP 的 low-rank 分布。
                "module_type": item.get("module_type", "unknown"),
                # up / down 都先转成 half 并搬到 CPU，减少保存文件体积。
                "up": up.half().cpu(),
                "down": down.half().cpu(),
            }
        )

    print("Low-rank states built.")
    # 返回所有选中层的 low-rank 状态。
    return low_rank_results


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
            if isinstance(m, nn.Linear) and any(
                [
                    _ in n
                    for _ in [
                        "wo",
                        "w2",
                        "down_proj",
                        "o_proj",
                        "v_proj",
                        "gate_proj",
                        "up_proj",
                        "w1",
                        "w3",
                    ]
                ]
            ):
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
                "cap_avg_grad": mean_cap.item(),
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
        model.language_model.model.tok_embeddings = (
            model.language_model.model.tok_embeddings.to(device)
        )
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


def _compute_reweight_medians(grad_avg_dict):
    attn_list = []
    mlp_list = []

    for key_name in grad_avg_dict:
        if "down_" in key_name or "w2" in key_name:
            mlp_list.append(
                grad_avg_dict[key_name]["vis_avg_grad"]
                / grad_avg_dict[key_name]["cap_avg_grad"]
            )
        if "o_proj" in key_name or "wo" in key_name:
            attn_list.append(
                grad_avg_dict[key_name]["vis_avg_grad"]
                / grad_avg_dict[key_name]["cap_avg_grad"]
            )

    return float(np.median(attn_list)), float(np.median(mlp_list))


def _load_reweight_cache(reweight_cache_path: Optional[str]):
    if not reweight_cache_path or not os.path.exists(reweight_cache_path):
        return None

    try:
        return torch.load(
            reweight_cache_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(reweight_cache_path, map_location="cpu")


def _save_reweight_cache(reweight_cache_path: Optional[str], reweight_cache: dict):
    if not reweight_cache_path:
        return

    dirpath = os.path.dirname(reweight_cache_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    torch.save(reweight_cache, reweight_cache_path)


@torch.no_grad()
def run_mbq(
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit,  # 新增
    q_config,
    auto_scale=True,
    # 以下参数都新增
    loss_mode="mae",
    wa_quant=False,
    reweight=False,
    reweight_cache_path: Optional[str] = None,
    distort=False,
    use_low_rank=False,
    low_rank_rank=16,
    low_rank_attn_topk_ratio=0.4,
    low_rank_mlp_topk_ratio=0.4,
    linear_mixed_probe=False,
    linear_probe_high_bit=4,
    linear_probe_keep_ratio=0.5,
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
    except ValueError:  # work with early exit
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
        "low_rank": [],
        "linear_score_map": {},
        "linear_bit_map": {},
    }
    low_rank_candidates = []
    linear_score_entries = []

    # ===========MBQ:下面reweight和distort都是新增
    if reweight:
        reweight_cache = _load_reweight_cache(reweight_cache_path)

        if reweight_cache is not None:
            grad_avg_dict = reweight_cache["grad_avg_dict"]
            attn_median = reweight_cache.get("attn_median")
            mlp_median = reweight_cache.get("mlp_median")
            if attn_median is None or mlp_median is None:
                attn_median, mlp_median = _compute_reweight_medians(grad_avg_dict)
                reweight_cache["attn_median"] = attn_median
                reweight_cache["mlp_median"] = mlp_median
            print("Loaded reweight cache, skipping gradient recomputation.")
        else:
            model.to_cuda()
            print("Save gradient...")
            # save gradient
            grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask)
            grad_cache.register_hooks(layers=layers)

            with torch.enable_grad():
                mini_batch = 1
                total_samples = next(iter(prompt_inputs.values())).shape[0]
                accum_steps = int(total_samples / mini_batch)

                for i in tqdm.tqdm(
                    range(0, total_samples, mini_batch),
                    desc="Running gradient calculation...",
                ):
                    mini_inputs = {}
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            mini_inputs[k] = inputs[k][i : i + mini_batch]

                    outputs = model(**mini_inputs)

                    loss = outputs[0]

                    loss = loss / accum_steps
                    loss.backward()

            model.to_cpu()
            grad_avg_dict = grad_cache.get_avg_grad_dict()
            grad_cache.remove_hooks()
            del grad_cache

            attn_median, mlp_median = _compute_reweight_medians(grad_avg_dict)
            reweight_cache = {
                "grad_avg_dict": grad_avg_dict,
                "attn_median": float(attn_median),
                "mlp_median": float(mlp_median),
            }

            _save_reweight_cache(reweight_cache_path, reweight_cache)
            if reweight_cache_path:
                print("Reweight cache saved at", reweight_cache_path)

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
        setattr(layer, "_mbq_full_name", get_op_name(model.model, layer))
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

        # =======新增
        if reweight:
            scale_reweight_ratio_dict = {}
            for key, value in grad_avg_dict.items():
                item_list = key.split(".")
                if str(i) in item_list:
                    if "wo" in item_list or "o_proj" in item_list:
                        scale_reweight_ratio_dict["attn"] = max(
                            (value["vis_avg_grad"] / value["cap_avg_grad"]), attn_median
                        )
                    elif "w2" in item_list or "down_proj" in item_list:
                        scale_reweight_ratio_dict["mlp"] = max(
                            (value["vis_avg_grad"] / value["cap_avg_grad"]), mlp_median
                        )
        else:
            scale_reweight_ratio_dict = {"attn": None, "mlp": None}
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
                        loss_mode=loss_mode,
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
                        loss_mode=loss_mode,
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
                        loss_mode=loss_mode,
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
                        loss_mode=loss_mode,
                    )

            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

            if (use_low_rank or linear_mixed_probe) and (not wa_quant):
                layer_name = get_op_name(model.model, layer)
                layer_linear_scores = _collect_internvl2_linear_scores(
                    layer=layer,
                    layer_name=layer_name,
                    input_feat=input_feat,
                    w_bit=w_bit,
                    q_config=q_config,
                    ans_mask=caption_mask,
                    vis_mask=vision_mask,
                    reweight_ratio_dict=scale_reweight_ratio_dict,
                    emit_timing=use_low_rank,
                )
                if use_low_rank:
                    low_rank_candidates.extend(
                        [
                            {**item, "rank": int(low_rank_rank)}
                            for item in layer_linear_scores
                        ]
                    )
                if linear_mixed_probe:
                    linear_score_entries.extend(layer_linear_scores)

            # =========新增
            if distort:
                # get distort output as next layer's input
                if wa_quant:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        new_linear = WALinear.from_float(
                            m,
                            weight_quant="per_channel",
                            act_quant="per_token",
                            w_bit=w_bit,
                            a_bit=a_bit,
                        )
                        father_module = get_module_by_name_suffix(
                            layer_q, ".".join(n.split(".")[:-1])
                        )
                        setattr(father_module, n.split(".")[-1], new_linear)
                        del new_linear, m
                        torch.cuda.empty_cache()

                    inps_distort = inps_distort.to(
                        next(layer_q.parameters()).device
                    )  # in case multi-gpu
                    inps_distort = layer_q(inps_distort, **layer_kwargs)[0]
                    del layer_q
                else:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        m.weight.data = pseudo_quantize_tensor(
                            m.weight.data, n_bits=w_bit, **q_config
                        )
                        torch.cuda.empty_cache()

                    inps_distort = inps_distort.to(
                        next(layer_q.parameters()).device
                    )  # in case multi-gpu
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

    if use_low_rank and (not wa_quant):
        print("Building low-rank states...")
        mbq_results["low_rank"] = _build_low_rank_states(
            model.model,
            low_rank_candidates,
            w_bit,
            q_config,
            attn_topk_ratio=low_rank_attn_topk_ratio,
            mlp_topk_ratio=low_rank_mlp_topk_ratio,
        )

    if linear_mixed_probe and linear_score_entries:
        mbq_results["linear_score_map"] = {
            item["name"]: {
                "score": item["score"],
                "module_type": item["module_type"],
                "module_family": item["module_family"],
                "w_bit": item["w_bit"],
            }
            for item in linear_score_entries
        }
        mbq_results["linear_bit_map"] = _build_linear_bit_map(
            linear_score_entries,
            default_w_bit=w_bit,
            high_w_bit=linear_probe_high_bit,
            keep_ratio=linear_probe_keep_ratio,
        )

    return mbq_results


def apply_mbq(model, mbq_results):
    apply_scale(model, mbq_results["scale"])
