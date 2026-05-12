import os

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
def _collect_internvl2_low_rank_candidates(
    layer,
    layer_name,
    input_feat,
    w_bit,
    q_config,
    rank,
    ans_mask=None,
    vis_mask=None,
    reweight_ratio_dict=None,
):
    """收集 InternVL2 / InternLM2 的 low-rank 候选层。

    这个函数当前只负责“打分”，不真正计算 SVD。
    它会对当前 layer 中预定义的 attention / MLP 线性层做一次与真实部署一致的伪量化，
    然后根据权重残差 `W - W_q` 的相对大小，得到这些层是否值得进入 top-k 的分数。

    参数:
        layer: 当前正在处理的 transformer block。
        layer_name: 当前 block 在完整模型中的路径名。
        input_feat: 当前 block 缓存到的各线性层输入特征，当前函数里用于判断目标模块是否存在可用输入。
        w_bit: 权重量化 bit 数。
        q_config: 当前量化配置，如 group size、zero point 等。
        rank: 如果该层后续被选中，预期分配给它的固定 low-rank rank。

    返回:
        candidates: 一个列表；每个字典表示当前层里一个可用于 low-rank 补偿的候选线性层。
    """
    # 初始化候选列表；保持列表接口便于把 attention 和 MLP 一起统一排序选 top-k。
    candidates = []
    # 只在 InternLM2DecoderLayer 上启用当前逻辑，避免误作用到其他架构。
    if layer.__class__.__name__ != "InternLM2DecoderLayer":
        return candidates
    # (module_name, module_type, module_family)
    # module_family 同时用作 reweight_key，因为当前 attention / MLP 的权重系数也是 attn / mlp。
    candidate_specs = [
        ("attention.wqkv", "attention_wqkv", "attn"),
        ("feed_forward.w1", "mlp_w1", "mlp"),
        ("feed_forward.w3", "mlp_w3", "mlp"),
        ("feed_forward.w2", "mlp_w2", "mlp"),
    ]

    def _compute_activation_aware_relative_error(
        flat_input,
        weight_fp,
        residual_weight,
        token_mask=None,
        chunk_size=1024,
    ):
        # 如果传入了 token_mask，只保留被选中的 token 的激活值。
        if token_mask is not None:
            token_mask = token_mask.reshape(-1).to(dtype=torch.bool)
            # 如果 mask 形状与 token 总数不匹配，说明 mask 与输入不兼容，直接返回 None。
            if token_mask.numel() != flat_input.shape[0]:
                return None
            # 用 mask 筛选出有效的激活值（例如只保留 answer 或 vision 区域）。
            flat_input = flat_input[token_mask]

        # 如果筛选后没有剩余 token，无法计算相对误差，返回 None。
        if flat_input.numel() == 0:
            return None

        # 把计算放到权重所在的设备上，避免反复搬运。
        device = weight_fp.device
        # numerator 累积量化后的输出误差能量：||X(W - Wq)||²。
        numerator = torch.zeros((), dtype=torch.float32, device=device)
        # denominator 累积原始输出能量：||XW||²，用于归一化。
        denominator = torch.zeros((), dtype=torch.float32, device=device)
        # 分块计算，避免单次矩阵乘法占用过多显存。
        for start in range(0, flat_input.shape[0], chunk_size):
            feat_chunk = flat_input[start : start + chunk_size].to(
                device, non_blocking=True
            )
            ref_chunk = torch.nn.functional.linear(feat_chunk, weight_fp)
            err_chunk = torch.nn.functional.linear(feat_chunk, residual_weight)
            numerator += err_chunk.float().pow(2).sum()
            denominator += ref_chunk.float().pow(2).sum()

        # 返回相对误差：||X(W - Wq)||² / ||XW||²。
        # clamp 防止分母接近 0 导致除零异常。
        return float((numerator / denominator.clamp(min=1e-6)).item())

    def _compute_multimodal_activation_aware_score(
        module_input,
        weight_fp,
        weight_q,
        token_ans_mask=None,
        token_vis_mask=None,
        reweight_ratio=None,
    ):
        # 将输入激活展平为 [所有 token, hidden_dim]，并转为 float 保证精度。
        flat_input = module_input.reshape(-1, module_input.shape[-1]).float()
        # 计算量化造成的权重残差矩阵：W - Wq。
        residual_weight = (weight_fp - weight_q).float()

        # 先计算全局（所有 token）的激活感知相对误差。
        global_score = _compute_activation_aware_relative_error(
            flat_input,
            weight_fp,
            residual_weight,
        )
        # 如果没有多模态 mask（纯文本场景），直接返回全局误差作为得分。
        if token_ans_mask is None or token_vis_mask is None:
            return global_score

        # 计算只在 answer（caption）token 上的激活感知相对误差。
        # 这衡量了该层量化对文本输出质量的直接影响。
        ans_score = _compute_activation_aware_relative_error(
            flat_input,
            weight_fp,
            residual_weight,
            token_mask=token_ans_mask,
        )
        # 计算只在视觉 token 上的激活感知相对误差。
        # 这衡量了该层量化对视觉理解质量的直接影响。
        vis_score = _compute_activation_aware_relative_error(
            flat_input,
            weight_fp,
            residual_weight,
            token_mask=token_vis_mask,
        )

        # 如果某个区域的 mask 筛选后没有可用 token，就用另一个区域的得分代替。
        if ans_score is None and vis_score is None:
            return global_score
        if ans_score is None:
            return vis_score
        if vis_score is None:
            return ans_score
        # 如果没有 reweight 系数，直接简单相加。
        if reweight_ratio is None:
            return ans_score + vis_score
        # 用 reweight 系数加权合并：L = L_ans + lambda * L_vis。
        # lambda 越大，视觉区域的量化误差在候选评分中权重越高。
        return ans_score + float(reweight_ratio) * vis_score

    for module_name, module_type, module_family in candidate_specs:
        # 当前前向过程中没有缓存到该线性层输入特征，说明这条路径当前不可用，直接跳过。
        if module_name not in input_feat:
            continue

        module = get_op_by_name(layer, module_name)
        if module is None or not isinstance(module, nn.Linear):
            continue

        # 读取 apply_scale 之后的浮点权重；这里 detach + float，是为了后续稳定做误差评估。
        weight_fp = module.weight.data.detach().float()
        # 用与真实部署一致的量化配置做一次伪量化，得到 2bit / 4bit 等主路径上的量化权重。
        weight_q = pseudo_quantize_tensor(
            weight_fp, n_bits=w_bit, inplace=False, **q_config
        )
        reweight_ratio = None
        if reweight_ratio_dict is not None:
            reweight_ratio = reweight_ratio_dict.get(module_family)

        # 分数越大，表示该层在真实多模态激活下量化后丢失的信息越多，越值得进入后续 top-k low-rank 补偿。
        score = _compute_multimodal_activation_aware_score(
            input_feat[module_name],
            weight_fp,
            weight_q,
            token_ans_mask=ans_mask,
            token_vis_mask=vis_mask,
            reweight_ratio=reweight_ratio,
        )

        # 把当前层登记为候选项；这里只保存轻量信息，真正的 SVD 会在 top-k 选完后再统一做。
        candidates.append(
            {
                # 保存完整模块路径，后续构建 low-rank 状态时会据此重新拿回真实模块。
                "name": layer_name + "." + module_name,
                # 保存当前层的量化残差分数，供后续排序选 top-k。
                "score": float(score.item()),
                # 当前版本先使用固定 rank；真正构建状态时仍会再和最大可分解 rank 取 min。
                "rank": int(rank),
                # 保存模块类型，便于后续打印 attention / MLP 的候选与入选统计。
                "module_type": module_type,
                # 保存模块族，供 attention / MLP 分桶 top-k 使用。
                "module_family": module_family,
            }
        )

    # 返回当前层收集到的候选结果。
    return candidates


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
        return sorted(group_candidates, key=lambda item: item["score"], reverse=True)[
            :topk_count
        ]

    attn_ratio = float(attn_topk_ratio)
    mlp_ratio = float(mlp_topk_ratio)
    attn_candidates = [
        item for item in candidates if item.get("module_family") == "attn"
    ]
    mlp_candidates = [item for item in candidates if item.get("module_family") == "mlp"]

    selected = []
    selected.extend(_take_topk(attn_candidates, attn_ratio))
    selected.extend(_take_topk(mlp_candidates, mlp_ratio))
    return sorted(selected, key=lambda item: item["score"], reverse=True)


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
    }
    low_rank_candidates = []

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

            if use_low_rank and (not wa_quant):
                low_rank_candidates.extend(
                    _collect_internvl2_low_rank_candidates(
                        layer,
                        get_op_name(model.model, layer),
                        input_feat,
                        w_bit,
                        q_config,
                        low_rank_rank,
                        ans_mask=caption_mask,
                        vis_mask=vision_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                    )
                )

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

    return mbq_results


def apply_mbq(model, mbq_results):
    apply_scale(model, mbq_results["scale"])
