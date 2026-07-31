"""独立脚本：为 $c_k$ 分析收集激活、残差 SVD 与模态 mask。

背景
----
MARS-VL 的 low-rank 残差补偿用普通 top-r SVD 分解 W2 残差
$R_m = W - Q(W)$。要判断 SVD 的能量序方向是否与「跨模态激活条件下的
输出误差方向」一致，需要计算

    c_k = sigma_k^2 * (
        ||X^{Omega_t} v_k||^2 / D_m(Omega_t)
        + rho_{g(m)} ||X^{Omega_v} v_k||^2 / D_m(Omega_v)
    )

其中 sigma_k, v_k 来自 $R_m$ 的 SVD，X^{Omega_t}/X^{Omega_v} 是校准文本 / 视觉
token 上的输入激活，D_m 是对应全精度输出能量，rho_{g(m)} 是 reweight 比例。

本脚本复用 `1_generate_scale.sh` 同款的 calibration forward（保证 mask 与
激活同源），但**跳过 scale grid search 与 SVD 因子构建**——scale 直接从已有
cache 加载并 apply，激活只跑一遍逐层 forward 收集。

输出（每个被选中做 low-rank 的 module 一个子目录）：
    {output_dir}/{module_idx}_{module_short}/
        meta.json     : name / bit / family / rho / shapes
        sigma.pt      : [min(in,out)] 奇异值（能量降序）
        component_stats.pt : 每个奇异方向上的 text / visual 激活能量
        svd_factor_prefix.pt : 与统计量同一 SVD 的可部署因子前缀
    {output_dir}/_global_meta.json : 全局配置 / rho 字典 / module 索引表

默认只保存计算和部署自适应 rank 所需的紧凑统计量与因子前缀；传入
``--save-raw-tensors`` 时才额外保存 X.pt / masks.pt / V.pt，供逐方向调试。
两种模式都可以在后续离线脚本里覆盖 rho，无需重跑 forward。

用法
----
python tools/collect_act_for_ck.py \\
    --config configs/internvl2/MBQ_search/my_8b_weight_only_svd.yaml \\
    --scale_path scale_cache/mbq/internvl2_8b_w2g32_scale_reweight_true_svd_1.0_custom_wo_all_w2_w3_48.pt \\
    --reweight_cache scale_cache/mbq/reweight/internvl2_8b_reweight_group.pt \\
    --output_dir act_for_ck/wo_all_w2_w3_48
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

# Support the documented ``python tools/collect_act_for_ck.py`` invocation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- 复用现有 MBQ / 模型代码 -------------------------------------------------
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.methods.mbq.quantize.auto_scale import apply_scale
from qmllm.methods.mbq.quantize.pre_quant import (
    get_blocks,
    get_named_linears,
    get_op_by_name,
    move_embed,
    process_input,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _short_name(full_name: str) -> str:
    """把 `language_model.model.layers.24.feed_forward.w3` 压成 `L24.w3`，
    方便作为目录名且保持可读。"""
    m = re.search(r"layers\.(\d+)\.(.+)$", full_name)
    if not m:
        return full_name.replace(".", "_")
    layer_idx, tail = m.group(1), m.group(2)
    return f"L{layer_idx}.{tail.replace('.', '_')}"


def _module_family(short: str) -> str:
    """映射到 reweight group family：attn_in/attn_out/mlp_in/mlp_out。

    与 pre_quant.py:1151-1170 的分组逻辑保持一致。"""
    if "wqkv" in short or "q_proj" in short or "k_proj" in short or "v_proj" in short:
        return "attn_in"
    if "wo" in short or "o_proj" in short:
        return "attn_out"
    if "w2" in short or "down_proj" in short:
        return "mlp_out"
    if "w1" in short or "w3" in short or "gate_proj" in short or "up_proj" in short:
        return "mlp_in"
    return "unknown"


def _tail_family(name: str) -> str:
    """Return a model-independent projection tail used by CLI filters."""
    tail = name.rsplit(".", 1)[-1]
    aliases = {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
    }
    return aliases.get(tail, tail)


def _matches_family(name: str, requested) -> bool:
    """Match either a concrete tail (wqkv/q_proj/...) or a role family."""
    if not requested:
        return True
    tail = _tail_family(name)
    role = _module_family(name)
    return tail in requested or role in requested


def _layer_index_of(full_name: str):
    m = re.search(r"layers\.(\d+)", full_name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 逐层 forward + hook 收集（仿 pre_quant.py:1090-1124，但只收 X，跳过 search）
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect_layer_inputs_once(layer, named_linears, inps, layer_kwargs):
    """Collect one batch slice. Kept separate so microbatching is exact."""
    feats = {}

    def hook(m, x, y, name):
        # The decoder output must stay on the accelerator for the next layer,
        # but the collected calibration rows are only consumed after this
        # forward.  Moving them to CPU here prevents every hooked projection
        # from retaining a full activation matrix on the GPU.
        feats[name] = x[0].reshape(-1, x[0].shape[-1]).detach().cpu()

    handles = [
        m.register_forward_hook(lambda mod, inp, out, n=name: hook(mod, inp, out, n))
        for name, m in named_linears.items()
    ]
    try:
        out = layer(inps, **layer_kwargs)[0]
    finally:
        for h in handles:
            h.remove()
    return feats, out


def _slice_layer_kwargs(layer_kwargs, total_batch, start, end):
    return {
        key: (value[start:end] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == total_batch else value)
        for key, value in layer_kwargs.items()
    }


@torch.no_grad()
def forward_layer(layer, inps, layer_kwargs, layer_batch_size=None):
    """Forward a decoder layer, optionally splitting only across batch items."""
    if layer_batch_size is None or inps.shape[0] <= layer_batch_size:
        return layer(inps, **layer_kwargs)[0]

    total_batch = inps.shape[0]
    chunks = []
    for start in range(0, total_batch, layer_batch_size):
        end = min(start + layer_batch_size, total_batch)
        chunks.append(layer(
            inps[start:end],
            **_slice_layer_kwargs(layer_kwargs, total_batch, start, end),
        )[0])
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def collect_layer_inputs(layer, named_linears, inps, layer_kwargs, layer_batch_size=None):
    """跑一次 layer forward，用 hook 抓每个 Linear 的输入，返回
    {name: [num_tokens, in_dim]}（已 cat、已搬 CPU）。"""
    if layer_batch_size is None or inps.shape[0] <= layer_batch_size:
        return _collect_layer_inputs_once(layer, named_linears, inps, layer_kwargs)

    feat_chunks = defaultdict(list)
    out_chunks = []
    total_batch = inps.shape[0]
    for start in range(0, total_batch, layer_batch_size):
        end = min(start + layer_batch_size, total_batch)
        chunk_feats, chunk_out = _collect_layer_inputs_once(
            layer,
            named_linears,
            inps[start:end],
            _slice_layer_kwargs(layer_kwargs, total_batch, start, end),
        )
        for name, value in chunk_feats.items():
            feat_chunks[name].append(value)
        out_chunks.append(chunk_out)

    return (
        {name: torch.cat(chunks, dim=0) for name, chunks in feat_chunks.items()},
        torch.cat(out_chunks, dim=0),
    )


def _pseudo_quant_with_q_config(weight_fp, w_bit, q_config):
    """按 MBQ 的 q_config 对单个权重做伪量化，返回 Q(W)。"""
    return pseudo_quantize_tensor(
        weight_fp,
        n_bits=w_bit,
        inplace=False,
        zero_point=q_config["zero_point"],
        q_group_size=q_config["q_group_size"],
        double_quant=q_config.get("double_quant", False),
        double_quant_config=q_config.get("double_quant_config") or {},
    )


@torch.no_grad()
def _squared_linear_output_pair(
    input_rows, reference_weight, residual_weight, device, chunk_size=256
):
    """Compute reference/residual output energies with one pair of transfers."""
    if input_rows.numel() == 0:
        return 0.0, 0.0
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    reference_device = reference_weight.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    residual_device = residual_weight.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    reference_total = torch.zeros((), dtype=torch.float64, device=device)
    residual_total = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, input_rows.shape[0], chunk_size):
        rows = input_rows[start : start + chunk_size].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        reference_output = torch.nn.functional.linear(rows, reference_device)
        residual_output = torch.nn.functional.linear(rows, residual_device)
        reference_total += reference_output.double().square().sum()
        residual_total += residual_output.double().square().sum()
        del rows, reference_output, residual_output
    values = float(reference_total.cpu()), float(residual_total.cpu())
    del reference_device, residual_device, reference_total, residual_total
    return values


@torch.no_grad()
def _squared_frobenius_norm(tensor, chunk_size=256):
    """Compute a stable squared Frobenius norm with bounded temporary memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rows = tensor.reshape(-1, tensor.shape[-1])
    total = torch.zeros((), dtype=torch.float64, device=rows.device)
    for start in range(0, rows.shape[0], chunk_size):
        total += rows[start : start + chunk_size].double().square().sum()
    return float(total.cpu())


@torch.no_grad()
def _component_input_energies(
    X,
    text_mask,
    vis_mask,
    right_vectors,
    device,
    chunk_size=256,
):
    """Return ||X_text v_k||^2 and ||X_vis v_k||^2 for every SVD direction."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    directions = right_vectors.to(
        device=device, dtype=torch.float32, non_blocking=True
    )

    def accumulate(input_rows):
        total = torch.zeros(
            directions.shape[0], dtype=torch.float64, device=device
        )
        for start in range(0, input_rows.shape[0], chunk_size):
            rows = input_rows[start : start + chunk_size].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            projected = torch.nn.functional.linear(rows, directions)
            total += projected.double().square().sum(dim=0)
            del rows, projected
        return total.cpu()

    text_energy = accumulate(X[text_mask])
    vis_energy = accumulate(X[vis_mask])
    del directions
    return text_energy, vis_energy


@torch.no_grad()
def _projection_error_statistics(
    X,
    text_mask,
    vis_mask,
    weight_fp,
    residual,
    rho,
    device,
    chunk_size,
    eps=1e-6,
):
    """Compute the paper's modality-normalized projection score E_m."""
    X_text = X[text_mask]
    X_vis = X[vis_mask]
    denominator_text, numerator_text = _squared_linear_output_pair(
        X_text, weight_fp, residual, device, chunk_size
    )
    denominator_vis, numerator_vis = _squared_linear_output_pair(
        X_vis, weight_fp, residual, device, chunk_size
    )
    denominator_text = max(denominator_text, eps)
    denominator_vis = max(denominator_vis, eps)
    score_text = numerator_text / denominator_text
    score_vis = numerator_vis / denominator_vis
    return {
        "denominator_text": denominator_text,
        "denominator_vis": denominator_vis,
        "numerator_text": numerator_text,
        "numerator_vis": numerator_vis,
        "score_text": score_text,
        "score_vis": score_vis,
        "projection_score": score_text + float(rho) * score_vis,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale_path", required=True)
    parser.add_argument(
        "--reweight_cache",
        default=None,
        help=(
            "Role-specific gradient-ratio cache. Defaults to reweight_cache_path "
            "from --config; a paper-aligned collection requires this cache."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--direct-local-loader",
        action="store_true",
        help="Load InternVL2 directly from the local Hugging Face snapshot instead "
        "of going through lmms-eval. Intended for an offline-compatible runtime.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="For --direct-local-loader, refuse all Hugging Face network access.",
    )
    parser.add_argument(
        "--only-family",
        nargs="+",
        default=None,
        help=(
            "Only collect selected projection tails or role families. Examples: "
            "wqkv wo q_proj o_proj attn_in mlp_out. Useful for supplemental "
            "collection without recomputing existing modules."
        ),
    )
    parser.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help=(
            "Allow an existing directory only when it has no prior "
            "_global_meta.json and no colliding module subdirectories. Existing "
            "collection files are never overwritten."
        ),
    )
    # 允许覆盖 config 里的样本数（少跑几张可以快速验证脚本）
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument(
        "--layer-batch-size",
        type=int,
        default=None,
        help="Microbatch size for decoder-layer forwards. Preserves activation "
        "order while reducing quadratic attention memory.",
    )
    # SVD 截断维度：None=完整 SVD；设大值用 svd_lowrank 加速
    parser.add_argument("--svd_full_max_dim", type=int, default=2048,
                        help="当 min(in,out) 超过此值时改用 torch.svd_lowrank 以省显存；"
                             "否则用 linalg.svd 拿完整谱。")
    parser.add_argument("--svd_q", type=int, default=512,
                        help="svd_lowrank 的探测维度 q。")
    parser.add_argument(
        "--factor-prefix-rank",
        type=int,
        default=128,
        help=(
            "Save the first N factors from the same SVD used for c_k. The "
            "adaptive cache builder uses these factors instead of truncating "
            "an independently computed SVD basis."
        ),
    )
    parser.add_argument(
        "--keep-modality-only",
        "--keep_modality_only",
        dest="keep_modality_only",
        action="store_true",
        default=True,
        help="只保留 supervised-text+visual token（默认开启）。",
    )
    parser.add_argument(
        "--no-keep-modality-only",
        "--no-keep_modality_only",
        dest="keep_modality_only",
        action="store_false",
        help="保留校准序列中的全部 token。",
    )
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="若 token 数超过此值，按 text/vis 等比例下采样截断（再省磁盘）。"
                             "None=不截断。建议 8192。")
    parser.add_argument(
        "--mask-semantics",
        choices=["all_text", "answer_text"],
        default="all_text",
        help=(
            "Definition of Omega_t. all_text (default) follows the paper and "
            "uses every valid non-visual text token; answer_text is retained "
            "only for diagnostics against the original MBQ caption mask."
        ),
    )
    parser.add_argument(
        "--energy-chunk-size",
        type=int,
        default=256,
        help="Token chunk size for exact modality output-energy normalizers.",
    )
    parser.add_argument(
        "--save-raw-tensors",
        action="store_true",
        help=(
            "Also save X.pt, masks.pt, and V.pt. Adaptive-rank analysis uses "
            "component_stats.pt and does not require these large tensors."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    reweight_cache_path = args.reweight_cache or cfg.get("reweight_cache_path")
    if not reweight_cache_path or not os.path.exists(reweight_cache_path):
        raise FileNotFoundError(
            "A role-specific reweight cache is required for paper-aligned c_k "
            f"collection, got {reweight_cache_path!r}"
        )

    n_samples = args.n_samples if args.n_samples is not None else cfg.get("n_samples", 128)
    if int(n_samples) < 1:
        raise ValueError("--n_samples must be positive")
    if args.layer_batch_size is not None and args.layer_batch_size < 1:
        raise ValueError("--layer-batch-size must be positive")
    if args.svd_full_max_dim < 1:
        raise ValueError("--svd-full-max-dim must be positive")
    if args.svd_q < 1:
        raise ValueError("--svd-q must be positive")
    if args.factor_prefix_rank < 1:
        raise ValueError("--factor-prefix-rank must be positive")
    if args.svd_q < args.factor_prefix_rank:
        raise ValueError("--svd-q must be at least --factor-prefix-rank")
    if args.energy_chunk_size < 1:
        raise ValueError("--energy-chunk-size must be positive")
    keep_modality_only = args.keep_modality_only
    max_tokens = args.max_tokens
    if max_tokens is not None and max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2 to retain both modalities")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if not args.allow_existing_output_dir:
            raise FileExistsError(
                f"Refusing to write into non-empty output directory: {args.output_dir}."
            )
        if os.path.exists(os.path.join(args.output_dir, "_global_meta.json")):
            raise FileExistsError(
                "Refusing to overwrite an existing activation collection; use a "
                "new --output_dir"
            )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. 加载模型（与 main_quant.py 同款流程）----------------------------
    if args.direct_local_loader:
        if cfg["model"] != "internvl2":
            raise ValueError(
                "--direct-local-loader currently supports only model=internvl2, "
                f"got {cfg['model']!r}"
            )
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from qmllm.models.internvl2.internvl2 import InternVL2

        model_args = dict(
            item.split("=", 1)
            for item in cfg["model_args"].split(",")
            if "=" in item
        )
        pretrained = model_args.get("pretrained")
        if not pretrained:
            raise ValueError("model_args must contain pretrained=<model-id>")
        loader_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "local_files_only": args.local_files_only,
        }
        # Avoid AutoModel's global mapping scan. Some offline environments have
        # an unrelated timm/Transformers mismatch in that scan, while the
        # cached InternVL2 remote class itself remains loadable.
        model_class = get_class_from_dynamic_module(
            "modeling_internvl_chat.InternVLChatModel",
            pretrained,
            local_files_only=args.local_files_only,
        )
        model = model_class.from_pretrained(pretrained, **loader_kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        process_model = InternVL2(model, tokenizer)
        print("[collect] loaded InternVL2 through direct local loader")
    else:
        from lmms_eval.models import get_model
        from qmllm.models import get_process_model

        ModelClass = get_model(cfg["model"])
        lm = ModelClass.create_from_arg_string(
            cfg["model_args"],
            {"batch_size": cfg.get("batch_size", 1), "device": cfg.get("device")},
        )
        ProcessClass = get_process_model(cfg["model"])
        process_model = ProcessClass(
            lm._model, lm._tokenizer,
            lm.processor if hasattr(lm, "processor") else None,
        )

    # ---- 2. 加载 cache + apply scale（拿到 scaled-but-not-quant 的权重）----
    mbq_results = torch.load(args.scale_path, map_location="cpu")
    scale_list = mbq_results["scale"]
    low_rank_entries = mbq_results.get("low_rank", [])
    linear_bit_map = mbq_results.get("linear_bit_map", {})

    # low_rank 列表里的 name 即「被选中做 SVD 补偿」的目标集合
    target_modules = [it["name"] for it in low_rank_entries]
    if args.only_family:
        wanted_families = set(args.only_family)
        target_modules = [
            name for name in target_modules
            if _matches_family(name, wanted_families)
        ]
    if not target_modules:
        raise ValueError(
            "No low-rank targets match --only-family="
            f"{args.only_family!r} in {args.scale_path}"
        )
    if len(set(target_modules)) != len(target_modules):
        raise ValueError("low_rank cache contains duplicate projection names")
    invalid_targets = [name for name in target_modules if _layer_index_of(name) is None]
    if invalid_targets:
        raise ValueError(
            "cannot parse decoder layer indices for low-rank projections: "
            f"{invalid_targets[:5]}"
        )
    print(
        f"[collect] target modules (low-rank): {len(target_modules)} "
        f"(family filter: {args.only_family or 'all'})"
    )

    # 按 layer 索引分组，方便逐层 forward 时只对目标层算 SVD
    targets_by_layer = {}
    for name in target_modules:
        li = _layer_index_of(name)
        targets_by_layer.setdefault(li, []).append(name)

    # apply scale 到 process_model.model（不 quant）
    apply_scale(process_model.model, scale_list)
    print("[collect] scale applied (weights are now scaled, not quantized)")

    if args.direct_local_loader:
        # generate_input only needs these modules; retaining all decoder layers
        # on CPU is equivalent to the later layerwise collector and avoids a
        # full 8B GPU residency solely to form calibration embeddings.
        process_model.model.language_model.get_input_embeddings().cuda()
        process_model.model.vision_model.cuda()
        process_model.model.mlp1.cuda()

    # ---- 3. 构建 calibration 数据 → 拿 vision_mask / caption_mask ----------
    prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(
        data_path=cfg["data_path"],
        image_folder=cfg["image_folder"],
        model=process_model,
        n_samples=n_samples,
        few_shot_format=cfg.get("few_shot_format", False),
        interleave_format=cfg.get("interleave_format", False),
        text_data_path=cfg.get("text_data_path", ""),
    )
    if args.direct_local_loader:
        process_model.model.language_model.get_input_embeddings().cpu()
        process_model.model.vision_model.cpu()
        process_model.model.mlp1.cpu()
        torch.cuda.empty_cache()
    # mask 留到 Catcher 段由 process_input pop，这里只先看一眼形状（不消费）
    print(f"[collect] prompt_kwargs keys: {list(prompt_kwargs.keys())}")

    # ---- 4. 逐层 forward 收集激活（仿 run_mbq 的 Catcher + 循环）----------
    # 注意：process_model 是 InternVL2 封装，process_model.model 是
    # InternVLChatModel。get_blocks / apply_scale / get_op_by_name 都作用在
    # process_model.model 上；to_cuda/to_cpu 作用在 process_model 上。
    layers = get_blocks(process_model.model)

    import torch.nn as nn
    import gc

    inps_list = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(process_model.model, "cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps_list.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])

    # 复用原版 process_input：合并 inputs、注入 use_cache=False、弹出两个 mask
    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)
    print(f"[collect] vision_mask {tuple(vision_mask.shape)}, "
          f"caption_mask {tuple(caption_mask.shape)}")
    # layer0 输入是 inputs_embeds，已含全部 token；mask 展平到 [num_tokens]。
    # 论文中的 Omega_t 在代码里显式记录语义，避免把 answer positions 和
    # all non-visual text positions 静默混用。
    if args.mask_semantics == "all_text":
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None or attention_mask.shape != vision_mask.shape:
            raise ValueError(
                "all_text mask semantics requires a 2-D attention_mask with the "
                "same shape as vision_mask"
            )
        text_mask = attention_mask.to(torch.bool) & ~vision_mask.to(torch.bool)
    else:
        text_mask = caption_mask.to(torch.bool)
    text_mask_flat = text_mask.reshape(-1).cpu()
    vis_mask_flat = vision_mask.reshape(-1).to(torch.bool).cpu()
    if torch.any(text_mask_flat & vis_mask_flat):
        raise ValueError("Omega_t and Omega_v must be disjoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import time as _time
    print(f"[collect] to_cuda start..."); _t0 = _time.time()
    if not args.direct_local_loader:
        process_model.to_cuda()
    print(f"[collect] to_cuda done ({_time.time()-_t0:.1f}s), "
          f"llm device={next(process_model.model.language_model.parameters()).device}")
    print(f"[collect] layer0 forward start..."); _t0 = _time.time()
    try:
        process_model(**inputs)
    except ValueError:
        pass
    print(f"[collect] layer0 forward done ({_time.time()-_t0:.1f}s)")
    if not args.direct_local_loader:
        process_model.to_cpu()

    layers[0] = layers[0].module  # 还原为原始 layer 0
    inps = inps_list[0]
    layer_kwargs["use_cache"] = False

    layers[0] = layers[0].cpu()
    move_embed(process_model.model, "cpu")
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[collect] layer-0 input embeds: {tuple(inps.shape)}")

    # 全局 rho 字典
    try:
        rw = torch.load(
            reweight_cache_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        rw = torch.load(reweight_cache_path, map_location="cpu")
    rho_dict = rw.get("reweight_medians", {})
    required_families = {_module_family(name) for name in target_modules}
    missing_rho = sorted(required_families - set(rho_dict))
    if missing_rho:
        raise ValueError(
            "reweight cache lacks paper-required role ratios for: "
            f"{missing_rho}"
        )
    print(f"[collect] rho (reweight_medians): {rho_dict}")

    # q_config：从 config 重建（与 mbq_entry 的 q_config 一致）
    q_config = {
        "zero_point": cfg.get("zero_point", True),
        "q_group_size": cfg.get("w_group", cfg.get("q_group_size", 128)),
        "double_quant": cfg.get("double_quant", False),
        "double_quant_config": cfg.get("double_quant_config") or {},
    }

    # ---- 5. 逐层：forward 收 X → 对目标 module 算 R_m 的 SVD → 落盘 ------
    module_index = []  # 记录 (idx, full_name, layer_idx, family, bit, rho)
    saved_count = 0

    for i in range(len(layers)):
        if i not in targets_by_layer:
            # 不是目标层，但仍要 forward 推进 inps（保持激活链路一致）
            layer = layers[i].to(device)
            for k in layer_kwargs:
                if isinstance(layer_kwargs[k], torch.Tensor):
                    layer_kwargs[k] = layer_kwargs[k].to(device)
            inps = forward_layer(
                layer,
                inps.to(device),
                layer_kwargs,
                layer_batch_size=args.layer_batch_size,
            )
            layers[i] = layers[i].to("cpu")
            torch.cuda.empty_cache()
            continue

        layer = layers[i].to(device)
        for k in layer_kwargs:
            if isinstance(layer_kwargs[k], torch.Tensor):
                layer_kwargs[k] = layer_kwargs[k].to(device)
        target_names = targets_by_layer[i]
        target_tails = {
            name.split(f"layers.{i}.", 1)[1]
            for name in target_names
        }
        all_named_linears = get_named_linears(layer)
        named_linears = {
            name: module
            for name, module in all_named_linears.items()
            if name in target_tails
        }
        missing_tails = sorted(target_tails - set(named_linears))
        if missing_tails:
            raise ValueError(
                f"target projections are absent from decoder layer {i}: {missing_tails}"
            )
        # Hook only the selected residual projections. Hooking every Linear can
        # retain several full activation matrices at once on large VLM layers.
        feats, inps = collect_layer_inputs(
            layer,
            named_linears,
            inps.to(device),
            layer_kwargs,
            layer_batch_size=args.layer_batch_size,
        )
        layers[i] = layers[i].to("cpu")
        torch.cuda.empty_cache()

        for full_name in target_names:
            # full_name = language_model.model.layers.{i}.{tail}
            tail = full_name.split(f"layers.{i}.", 1)[1]
            if tail not in feats:
                raise RuntimeError(f"failed to capture activations for {full_name}")
            X = feats[tail].float()  # [num_tokens, in_dim], already on CPU

            # 拿 scaled 权重，算 R_m = W_scaled - Q(W_scaled)
            module = get_op_by_name(process_model.model, full_name)
            if module is None:
                raise ValueError(f"cannot resolve target projection {full_name}")
            w_bit = int(linear_bit_map.get(full_name, cfg.get("w_bit", 2)))
            weight_fp = module.weight.data.detach().float()
            weight_q = _pseudo_quant_with_q_config(weight_fp, w_bit, q_config)
            residual = (weight_fp - weight_q).float()  # [out, in]

            # SVD：残差按能量排序。注意 R = U S V^T，我们要 sigma_k 与 V[:,k]
            min_dim = min(residual.shape)
            residual_gpu = residual.to(device)
            with torch.no_grad():
                if min_dim <= args.svd_full_max_dim:
                    U, S, Vh = torch.linalg.svd(residual_gpu, full_matrices=False)
                else:
                    # 大矩阵用随机 SVD。q 必须由调用方控制；此前这里把 q
                    # 错误地抬到 min_dim，导致 --svd_q 对大矩阵失效。
                    q = min(args.svd_q, min_dim)
                    if q < 1:
                        raise ValueError(f"--svd_q must be positive, got {args.svd_q}")
                    # svd_lowrank 期望 [n_features, k] 输入，转置一下
                    U, S, V = torch.svd_lowrank(residual_gpu, q=q, niter=4)
                    order = torch.argsort(S, descending=True)
                    S = S[order]
                    U = U[:, order]
                    Vh = V[:, order].transpose(0, 1)  # [min_dim, in]
            family = _module_family(tail)
            rho = float(rho_dict[family])

            # 模态过滤：c_k 只需 text+vis token，其余 token 零贡献，丢弃省磁盘
            keep_mask = text_mask_flat | vis_mask_flat
            if keep_modality_only:
                X_keep = X[keep_mask]
                text_sub = text_mask_flat[keep_mask]
                vis_sub = vis_mask_flat[keep_mask]
            else:
                X_keep = X
                text_sub = text_mask_flat
                vis_sub = vis_mask_flat

            # 可选截断：token 数超限则按 text/vis 等比例下采样
            n_keep = X_keep.shape[0]
            token_subsampled = False
            if max_tokens is not None and n_keep > max_tokens:
                token_subsampled = True
                text_idx = torch.nonzero(text_sub, as_tuple=False).flatten()
                vis_idx = torch.nonzero(vis_sub, as_tuple=False).flatten()
                n_text_t = int(round(max_tokens * text_idx.numel() / max(n_keep, 1)))
                n_vis_t = max_tokens - n_text_t
                text_pick = text_idx[torch.linspace(0, max(text_idx.numel()-1, 0), max(n_text_t, 0)).long()] if n_text_t > 0 and text_idx.numel() > 0 else text_idx.new_empty(0)
                vis_pick = vis_idx[torch.linspace(0, max(vis_idx.numel()-1, 0), max(n_vis_t, 0)).long()] if n_vis_t > 0 and vis_idx.numel() > 0 else vis_idx.new_empty(0)
                pick = torch.cat([text_pick, vis_pick]).unique()
                X_keep = X_keep[pick]
                text_sub = text_sub[pick]
                vis_sub = vis_sub[pick]

            if not torch.any(text_sub) or not torch.any(vis_sub):
                raise ValueError(
                    f"{full_name} has an empty supervised-text or visual mask "
                    "after token filtering"
                )

            error_stats = _projection_error_statistics(
                X=X_keep,
                text_mask=text_sub,
                vis_mask=vis_sub,
                weight_fp=weight_fp,
                residual=residual_gpu,
                rho=rho,
                device=device,
                chunk_size=args.energy_chunk_size,
            )
            component_text_energy, component_vis_energy = _component_input_energies(
                X=X_keep,
                text_mask=text_sub,
                vis_mask=vis_sub,
                right_vectors=Vh,
                device=device,
                chunk_size=args.energy_chunk_size,
            )
            residual_frobenius_sq = _squared_frobenius_norm(
                residual, chunk_size=args.energy_chunk_size
            )
            factor_prefix_rank = min(args.factor_prefix_rank, int(S.numel()))
            sqrt_s = torch.sqrt(S[:factor_prefix_rank])
            factor_up = (
                U[:, :factor_prefix_rank] * sqrt_s.unsqueeze(0)
            ).half().cpu()
            factor_down = (
                sqrt_s.unsqueeze(1) * Vh[:factor_prefix_rank, :]
            ).half().cpu()
            cached_score = (
                (mbq_results.get("linear_score_map") or {})
                .get(full_name, {})
                .get("score")
            )

            # 落盘
            sub = os.path.join(args.output_dir, f"{saved_count}_{_short_name(full_name)}")
            if os.path.exists(sub) and os.listdir(sub):
                raise FileExistsError(f"refusing to overwrite module output: {sub}")
            os.makedirs(sub, exist_ok=True)
            torch.save(S.float().cpu(), os.path.join(sub, "sigma.pt"))
            torch.save(
                {
                    "input_energy_text": component_text_energy,
                    "input_energy_vis": component_vis_energy,
                },
                os.path.join(sub, "component_stats.pt"),
            )
            # This unquantized prefix is derived from exactly the same ordered
            # decomposition as component_stats.pt.  The cache builder applies
            # the cache's factor-quantization configuration after choosing r_m.
            torch.save(
                {
                    "rank": factor_prefix_rank,
                    "up": factor_up,
                    "down": factor_down,
                    "storage_dtype": "float16",
                    "quantized": False,
                },
                os.path.join(sub, "svd_factor_prefix.pt"),
            )
            if args.save_raw_tensors:
                # Raw tensors are diagnostic-only; compact component statistics
                # are sufficient for all allocator calculations.
                torch.save(X_keep.half(), os.path.join(sub, "X.pt"))
                torch.save(
                    {"text": text_sub, "ans": text_sub, "vis": vis_sub},
                    os.path.join(sub, "masks.pt"),
                )
                torch.save(Vh.float().cpu().half(), os.path.join(sub, "V.pt"))
            with open(os.path.join(sub, "meta.json"), "w") as f:
                json.dump({
                    "name": full_name,
                    "short": _short_name(full_name),
                    "layer_idx": i,
                    "tail": tail,
                    "family": family,
                    "bit": w_bit,
                    "rho": rho,
                    "metric_version": 2,
                    "mask_semantics": args.mask_semantics,
                    "X_shape": list(X_keep.shape),
                    "sigma_len": int(S.numel()),
                    "V_shape": list(Vh.shape),
                    "weight_shape": list(residual.shape),
                    "factor_prefix_rank": factor_prefix_rank,
                    "factor_prefix_file": "svd_factor_prefix.pt",
                    "factor_prefix_basis": "collector_energy_ordered_svd_prefix",
                    "num_text_tokens": int(text_sub.sum()),
                    "num_ans_tokens": int(text_sub.sum()),
                    "num_vis_tokens": int(vis_sub.sum()),
                    "token_subsampled": token_subsampled,
                    "cached_projection_score": (
                        float(cached_score) if cached_score is not None else None
                    ),
                    "residual_frobenius_sq": residual_frobenius_sq,
                    "component_statistics": "input_energy_by_svd_direction",
                    "raw_tensors_saved": bool(args.save_raw_tensors),
                    **error_stats,
                }, f, indent=2)
            module_index.append({
                "idx": saved_count, "name": full_name,
                "short": _short_name(full_name), "layer_idx": i,
                "family": family, "bit": w_bit, "rho": rho,
                "metric_version": 2,
                "mask_semantics": args.mask_semantics,
                "projection_score": error_stats["projection_score"],
                "factor_prefix_rank": factor_prefix_rank,
                "token_subsampled": token_subsampled,
            })
            saved_count += 1
            del residual_gpu, U, S, Vh, factor_up, factor_down

        del feats
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 6. 全局 meta ----------------------------------------------------
    if saved_count != len(target_modules):
        saved_names = {item["name"] for item in module_index}
        missing_names = sorted(set(target_modules) - saved_names)
        raise RuntimeError(
            f"collected {saved_count}/{len(target_modules)} projections; "
            f"missing={missing_names[:5]}"
        )
    global_meta = {
        "config": str(Path(args.config).resolve()),
        "scale_path": str(Path(args.scale_path).resolve()),
        "reweight_cache": str(Path(reweight_cache_path).resolve()),
        "n_samples": n_samples,
        "layer_batch_size": args.layer_batch_size,
        "svd_full_max_dim": args.svd_full_max_dim,
        "svd_q": args.svd_q,
        "svd_niter": 4,
        "factor_prefix_rank": args.factor_prefix_rank,
        "factor_prefix_basis": "collector_energy_ordered_svd_prefix",
        "keep_modality_only": keep_modality_only,
        "max_tokens": max_tokens,
        "energy_chunk_size": args.energy_chunk_size,
        "save_raw_tensors": bool(args.save_raw_tensors),
        "metric_version": 2,
        "mask_semantics": args.mask_semantics,
        "only_family": args.only_family,
        "direct_local_loader": args.direct_local_loader,
        "local_files_only": args.local_files_only,
        "q_config": q_config,
        "rho_dict": rho_dict,
        "total_modules": saved_count,
        "module_index": module_index,
        "note": (
            "sigma and compact per-direction modality statistics are sufficient "
            "for c_k analysis. Per-modality reference and residual energies are "
            "stored in each meta.json; raw X/mask/V tensors are optional."
        ),
    }
    with open(os.path.join(args.output_dir, "_global_meta.json"), "w") as f:
        json.dump(global_meta, f, indent=2)

    print(f"[collect] done. saved {saved_count} modules to {args.output_dir}")


if __name__ == "__main__":
    main()
