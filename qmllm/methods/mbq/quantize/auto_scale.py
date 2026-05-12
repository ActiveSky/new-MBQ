import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from .qmodule import ScaledActivation
from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    """
    计算权重的逐通道重要性指标（per-channel weight importance）。

    用于评估每个 channel 内权重的"不均匀程度"：
      scale[c] = mean(|W[c,i]| / max(|W[c,:]|))
    值越接近 1 说明该 channel 内权重分布越均匀，
    值越小说明该 channel 内有少数"主导"权重。

    参数:
        weight:        权重矩阵，形状 [out_features, in_features]
        q_group_size:  分组量化的组大小，>0 时先将 weight reshape 为分组维度

    返回:
        scale: 形状 [in_features] 的逐通道重要性向量
    """
    org_shape = weight.shape
    if q_group_size > 0:
        # 分组量化：[-1, q_group_size]，每组内独立归一化
        weight = weight.view(-1, q_group_size)
    # 每个元素除以所在行的最大值，得到 [0,1] 的相对权重
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)  # 恢复原始形状
    scale = scale.mean(0)  # 沿 out_features 方向取均值 → 每个输入 channel 的重要性
    return scale


@torch.no_grad()
def get_act_scale(x):
    """
    计算激活值的逐通道"活跃度"（per-channel activation magnitude）。

    这是 AWQ/MBQ scale 搜索的核心输入——x_max 用于构造候选 scale。

    运算: x.abs().view(-1, hidden).mean(0)
      - view(-1, hidden): 将所有 token 展平到第一个维度，保留 hidden 维度
      - mean(0):          沿 token 维度取平均 → 形状 [hidden]

    直观理解: x_max[c] = 所有 token 在第 c 个 channel 上的平均激活绝对值。
    值越大的 channel 说明激活越"活跃",指导接下来对weight的scale的选择。
    AWQ 假设: 激活中越活跃的 channel 对输出越重要,因此倾向于给这些对应的weight的  channel 分配更大的 scale（更高精度）。

    参数:
        x: 激活值张量 [batch, seq_len, hidden] 或 [tokens, hidden]

    返回:
        形状 [hidden] 的逐通道平均激活幅度
    """
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """
    SmoothQuant / AWQ 核心重参数化：将 scale 从 LN 迁移到后续 FC 层的权重中。

    数学等价变换（Y = XW）：
      原始:  Y = LN(X) * W                    LN 输出直接进入 FC
      变换:  LN' = LN / scale               LN 权重缩小
             W'  = W * scale                FC 权重放大（等价于输入缩小）
             Y'  = LN'(X) * W' = LN(X) * W = Y  输出不变 ✓

    这样做的目的：把激活值的动态范围"压缩"到权重中，减少激活量化的误差。

    参数:
        ln:     LayerNorm 或 RMSNorm 层
        fcs:    后续的 Linear 层列表（一个 scale 可以同时影响多个 FC，如 QKV 或 gate+up）
        scales: 逐通道 scale 向量 [hidden_dim]
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)  # 确保 scale 和 LN 在同一设备

    # LN 权重除以 scale：减小 LN 输出 → 等价于后续 FC 的输入变小
    ln.weight.div_(scales)  # 原地修改，shape: [hidden] / [hidden] → [hidden]
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)  # bias 也要同步缩放

    # FC 权重乘以 scale：补偿 LN 输出的缩小，保持输出值不变
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))  # [out, hidden] * [1, hidden] → 广播到每行

    # NaN 检查：确保数值稳定性
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """
    将 scale 从前一个 FC 的输出通道迁移到后一个 FC 的输入通道。

    适用于: V_proj → O_proj, up_proj → down_proj 等 FC→FC 连接。

    数学等价变换：
      原始:  h = fc1(x)              [tokens, intermediate]
             y = fc2(h)              [tokens, hidden]
      变换:  fc1' = fc1 / scale      fc1 的 output 变小
             fc2' = fc2 * scale      fc2 的 weight 变大（补偿 fc1 输出变小）
             y'  = fc2'(fc1'(x)) = y 输出不变 ✓

    注意: fc1.weight[-scales.size(0):].div_(scales)
      InternLM2 的 wqkv 是 fused 的（Q+K+V 拼在一起），输出维度 = 3 × hidden，
      但 scale 只对应 Q 或 V 的部分（hidden 大小）。所以只对末尾 hidden 行做除法。

    参数:
        fc1:    前一个 Linear 层
        fc2:    后一个 Linear 层
        scales: 逐通道 scale 向量 [fc2.in_features] 或 [hidden]
    """
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    # fc1 输出通道除以 scale（只影响末尾的 hidden 行，兼容 fused QKV）
    # fc1.weight 形状 [out_features, in_features]
    # fc1.weight[-scales.size(0):] 取出最后 hidden 行 → [hidden, in_features]
    fc1.weight[-scales.size(0) :].div_(
        scales.view(-1, 1)
    )  # [hidden, in] / [hidden, 1] → 广播到每列
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))  # bias 也要同步缩放

    # fc2 输入通道乘以 scale：补偿 fc1 输出变小
    # fc2.weight 形状 [out, hidden]，scales.view(1, -1) → [1, hidden] 广播到每行
    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    """
    将 scale 从 GELU 激活函数迁移到后续 FC 层（Bloom/OPT 模型专用）。

    不同于 scale_ln_fcs（LN 有可学习的 weight 可以缩放），
    GELU 是纯函数没有参数，所以 scale 直接施加到 FC 权重上。

    注意：这里只有 fc.weight *= scale，没有 GELU 侧的除法——
    因为 GELU 的输出是动态值，无法通过权重复参数化来"抵消"。
    这是近似处理，但实践中效果可接受。

    参数:
        gelu:   GELU 或 BloomGelu 激活函数
        fc:     后续的 Linear 层
        scales: 逐通道 scale 向量
    """
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


# MBQ新增的参数：vis_mask, reweight_ratio_dict, loss_mode="mae"
# 内部函数几乎都增加了 reweight_ratio参数，其他没怎么变化
@torch.no_grad()
def auto_scale_block(
    module,
    module_kwargs,
    w_bit,
    q_config,
    input_feat,
    ans_mask,
    vis_mask,
    reweight_ratio_dict,
    loss_mode="mae",
):
    """
    MBQ 核心：对单个 transformer layer 自动搜索最优的逐通道 scale 因子。

    整体流程：
      对 attention 和 MLP 各自的两组权重（LN→FC 和 FC→FC），
      在 50 个候选 ratio 中网格搜索，找到使量化后输出误差最小的 scale。

    参数:
        module:               当前 transformer layer（如 InternLM2DecoderLayer）
        module_kwargs:        layer forward 的额外 kwargs（attention_mask, position_ids 等）
        w_bit:                权重量化位宽，None 表示不量化（identity）
        q_config:             量化配置 dict（zero_point, q_group_size）
        input_feat:           该层所有 Linear 的输入激活值 dict {layer_name: tensor}
        ans_mask:             答案 token 的 mask（多模态）
        vis_mask:             视觉 token 的 mask（多模态）
        reweight_ratio_dict:  {"attn": ratio, "mlp": ratio}，视觉/文本误差的权重
        loss_mode:            "mse" 或 "mae"

    返回:
        scales_list: list of (prev_op_name, (fc_names,), scale_tensor)
    """

    # ---- 构建权重量化函数 ----
    # 如果指定了 w_bit，用伪量化函数模拟 int 精度损失；
    # 否则返回恒等函数（不模拟量化，只做 scale 搜索）
    if w_bit is not None:

        def w_quantize_func(p):
            # 模拟量化：float → int(n_bits) → float
            return pseudo_quantize_tensor(
                p,
                n_bits=w_bit,
                **q_config,
            ).detach()

    else:

        def w_quantize_func(p):
            return p  # 不模拟量化，直接返回原始权重

    # 移除 use_cache，因为校准不需要 KV cache
    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # ============================================================
    # 内部函数：对一组 Linear 层搜索最优 scale
    # ============================================================
    def _search_module_scale(
        block, linears2scale: list, x, reweight_ratio=None, kwargs={}
    ):
        """
        对 block 内的 linears2scale 列表搜索最优 per-channel scale。

        算法：网格搜索 50 个候选 ratio ∈ [0, 1]
          对每个 ratio:
            1. scales = x_max^ratio  （基于激活统计构造 scale）
            2. 临时给 weight 施加 scale → 伪量化 → 除以 scale 恢复量级
            3. 用修改后的权重跑 block forward，计算与原始输出的误差
            4. 恢复原始 state_dict
          选误差最小的 ratio 对应的 scales

        参数:
            block:           被检查的模块（如 attention 或 mlp 子模块）
            linears2scale:   要施加 scale 的 Linear 层列表
            x:               输入激活值 [tokens, hidden_dim]
            reweight_ratio:  视觉 token 误差的权重系数
            kwargs:          block forward 的额外参数

        返回:
            best_scales: 形状 [hidden_dim] 的逐通道最优 scale
        """
        # w: co, ci  (权重矩阵: 输出维度 × 输入维度)
        # x: n, ci   (激活值: token数 × 输入维度)
        x = x.to(next(block.parameters()).device)  # 确保 x 和 block 在同一设备

        # 记录原始输出（未量化时的 ground truth）
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        # 基于激活值计算 x_max，用于构造候选 scale
        # get_act_scale: x.abs().view(-1, hidden).mean(0) → [hidden]
        #   即每个 channel 的平均绝对值，反映该 channel 的激活"活跃度"
        x_max = get_act_scale(x)

        best_error = float("inf")  # 当前最优误差
        best_ratio = -1  # 最优 ratio
        best_scales = None  # 最优 scale 张量

        # n_grid = 20
        n_grid = 50  # 第一阶段：全局粗搜 50 档（尤其对 2bit 量化）
        local_refine_grid = 11  # 第二阶段：在最优点附近再做局部细搜
        local_refine_half_span = 1 / n_grid
        history = []  # 记录所有 ratio 的 loss，用于调试

        # 保存原始 state_dict，每次迭代结束后恢复
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}

        def _evaluate_scale_ratio(ratio):
            # ---- 构造候选 scale ----
            # scales = x_max^ratio，ratio 越大 scale 越接近激活分布
            # clamp 防止极小值导致数值不稳定
            # 归一化使得 scales 的几何均值 ≈ 1，避免整体放大/缩小
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()  # 归一化

            # ---- 临时对权重施加 scale + 伪量化 ----
            # 流程：W' = Q(W * scale) / scale
            #   1. W *= scale      → 把 scale 迁移到权重
            #   2. Q(W')           → 模拟量化
            #   3. Q(W') / scale   → 恢复原始量级，但保留了量化误差
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))

            # 用修改后的权重跑 forward，得到量化后的输出
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            # ---- 计算量化误差 ----
            # 根据 loss_mode 和是否有 mask 选择不同的损失计算方式
            # out:   量化后的输出 [tokens, hidden]
            # org_out: 原始输出 [tokens, hidden]

            if loss_mode == "mse":
                # ----- MSE 模式：均方误差 -----
                if ans_mask is not None and vis_mask is not None:
                    # 多模态 + reweight：分别计算答案区域和视觉区域的 MSE
                    # ans_mask/vis_mask 形状 [tokens]，True 表示该 token 属于对应区域
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(
                        out
                    )  # [tokens] → [tokens, hidden]
                    vis_mask_expand = vis_mask.unsqueeze(-1).expand_as(out).cuda()
                    masked_diff_ans = (org_out - out).float().pow(2) * ans_mask_expand
                    masked_diff_vis = (org_out - out).float().pow(2) * vis_mask_expand
                    if reweight_ratio is not None:
                        # L = L_ans / N_ans + r * L_vis / N_vis
                        loss = (
                            masked_diff_ans.sum() / ans_mask_expand.sum()
                            + reweight_ratio
                            * (masked_diff_vis.sum() / vis_mask_expand.sum())
                        )
                    else:
                        loss = (org_out - out).float().pow(2).mean().item()
                elif ans_mask is not None and vis_mask is None:  # 和AWQ的代码一样
                    # AWQ 风格：只看答案 token 区域的 MSE
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                    masked_diff = (org_out - out).float().pow(2) * ans_mask_expand
                    loss = masked_diff.sum() / ans_mask_expand.sum()
                else:
                    # 无 mask：全局 MSE
                    loss = (
                        (org_out - out).float().pow(2).mean().item()
                    )  # float prevents overflow
            elif loss_mode == "mae":  # MBQ增加的loss计算方式
                # ----- MAE 模式：平均绝对误差（对离群值更鲁棒） -----
                if ans_mask is not None and vis_mask is not None:
                    # 多模态 + reweight：分别计算答案区域和视觉区域的 MAE
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                    vis_mask_expand = vis_mask.unsqueeze(-1).expand_as(out).cuda()
                    masked_diff_ans = (org_out - out).float().abs() * ans_mask_expand
                    masked_diff_vis = (org_out - out).float().abs() * vis_mask_expand
                    if reweight_ratio is not None:
                        # L = (L_ans + r * L_vis) / (N_ans + N_vis)
                        # 注意：MAE 模式下分母是总 token 数，不同于 MSE 的分别归一化
                        loss = (
                            masked_diff_ans.sum()
                            + reweight_ratio * masked_diff_vis.sum()
                        ) / (ans_mask_expand.sum() + vis_mask_expand.sum())
                    else:
                        loss = (org_out - out).float().abs().mean().item()
                elif ans_mask is not None and vis_mask is None:
                    # AWQ 风格：只看答案 token 区域的 MAE
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                    masked_diff = (org_out - out).float().abs() * ans_mask_expand
                    loss = masked_diff.sum() / ans_mask_expand.sum()
                else:
                    # 无 mask：全局 MAE
                    loss = (
                        (org_out - out).float().abs().mean().item()
                    )  # float prevents overflow

            # 恢复 block 的原始权重，准备下一个 ratio 的尝试
            block.load_state_dict(org_sd)
            return loss, scales

        for ratio in range(n_grid):
            # ratio ∈ {0, 0.02, 0.04, ..., 0.98}
            ratio = ratio * 1 / n_grid
            loss, scales = _evaluate_scale_ratio(ratio)

            # ---- 记录当前 ratio 的 loss 并追踪最优 ----
            history.append((ratio, loss))
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales

        # 第二阶段：在粗搜最优点附近做局部精细化搜索。
        # 判断条件：第一阶段已经找到有效最优 ratio，且局部细搜网格点数大于 1。
        if best_ratio != -1 and local_refine_grid > 1:
            # 计算局部搜索区间的左边界，确保不低于 0.0。
            refine_left = max(0.0, best_ratio - local_refine_half_span)
            # 计算局部搜索区间的右边界，确保不超过 1.0。
            refine_right = min(1.0, best_ratio + local_refine_half_span)
            # 在 [refine_left, refine_right] 区间内均匀生成 local_refine_grid 个候选 ratio。
            for ratio in torch.linspace(
                refine_left, refine_right, steps=local_refine_grid
            ).tolist():
                # 将 tensor 转为 Python 原生 float 类型。
                ratio = float(ratio)
                # 调用局部评估函数，计算当前 ratio 下的量化误差和对应的 scale 张量。
                loss, scales = _evaluate_scale_ratio(ratio)
                # 将当前 ratio 和 loss 记录到历史日志中，便于后续调试分析。
                history.append((ratio, loss))
                # 判断当前 loss 是否小于目前已知的最优误差。
                is_best = loss < best_error
                # 如果当前 ratio 更优，则更新最优误差、最优 ratio 和最优 scale。
                if is_best:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales

        # 安全检查：确保至少找到了一个有效的 ratio
        if best_ratio == -1:
            print(history)
            raise Exception

        # 展平为一维 [hidden_dim]
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # ============================================================
    # 内部函数：包装 _search_module_scale，返回带名称的格式化结果
    # ============================================================
    def _auto_get_scale(
        prev_op, layers, inp, reweight_ratio=None, module2inspect=None, kwargs={}
    ):
        """
        对一组 (prev_op → layers) 搜索最优 scale 并返回格式化结果。

        参数:
            prev_op:         前驱操作（LN 或 Linear）
            layers:          要施加 scale 的 Linear 层列表
            inp:             输入激活值
            reweight_ratio:  视觉 token 误差权重
            module2inspect:  用于检查输出误差的模块（默认取 layers[0]）
                             对 attention 和 mlp 通常传入整个子模块
            kwargs:          forward 的额外参数

        返回:
            (prev_op_name, (layer_names,), scales_tensor)
              名称是相对于当前 module 的路径，如 "attention_norm"
        """
        # module2inspect: 如果不指定，默认检查第一个 Linear 层的输出
        # 对 attention: 通常传入整个 self_attn，因为要检查 attention 整体输出
        # 对 mlp:     通常传入整个 mlp
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale(
            module2inspect, layers, inp, reweight_ratio, kwargs
        )
        scales = scales.detach().cpu()
        # 使用 get_op_name 获取相对于 module 的层内路径名
        return (
            get_op_name(module, prev_op),  # 如 "attention_norm"
            tuple([get_op_name(module, m) for m in layers]),  # 如 ("attention.wqkv",)
            scales,  # [hidden_dim] 的 scale 向量
        )

    scales_list = []  # 收集本层所有 scale 搜索结果

    # ============================================================
    # 模型类型分发：不同架构的层结构不同，需要不同的 prev_op/layers 组合
    # ============================================================

    if isinstance(module, OPTDecoderLayer):
        # -------------------- OPT 模型 --------------------
        # attention 输入: LN → Q,K,V
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attention 输出: V → O
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
            )
        )
        # MLP 输入: LN → FC1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
            )
        )
        # MLP 输出: FC1 → FC2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
            )
        )

    elif isinstance(module, LlamaDecoderLayer):
        # -------------------- Llama 模型 --------------------
        # attention 输入: input_layernorm → Q,K,V
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attention 输出: V → O
        # 条件：只有当 V 和 O 形状相同时才做 FC→FC scale
        # 参考: https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                )
            )
        # MLP 输入: post_attention_layernorm → gate_proj, up_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
            )
        )
        # MLP 输出: up_proj → down_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
            )
        )

    elif isinstance(module, BloomBlock):
        # -------------------- Bloom 模型 --------------------
        # attention 输入: LN → QKV (fused)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # Bloom 不做 attn out scale (QKV → dense)，因为 fused QKV 的分组不均
        # 参考: https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        # MLP 输入: LN → dense_h_to_4h
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # MLP 输出: GELU → dense_4h_to_h
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "mpt" in str(module.__class__).lower():
        # -------------------- MPT 模型 --------------------
        # attention 输入: norm_1 → Wqkv (fused)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1,
                layers=[module.attn.Wqkv],
                inp=input_feat["attn.Wqkv"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        # attention 输出: Wqkv → out_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv,
                layers=[module.attn.out_proj],
                inp=input_feat["attn.out_proj"],
            )
        )
        # MLP 输入: norm_2 → up_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2,
                layers=[module.ffn.up_proj],
                inp=input_feat["ffn.up_proj"],
                module2inspect=module.ffn,
            )
        )
        # MLP 输出: activation → down_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.act,
                layers=[module.ffn.down_proj],
                inp=input_feat["ffn.down_proj"],
            )
        )

    elif "falcon" in str(module.__class__).lower():
        # -------------------- Falcon 模型 --------------------
        # Falcon 不同尺寸的结构差异较大
        if "falcon-7b" in str(module.__class__).lower():
            # falcon-7b: 共享 LN，同时对 QKV 和 MLP FC1 做 scale
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.mlp.dense_h_to_4h,
                        module.self_attention.query_key_value,
                    ],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        elif "falcon-40b" in str(module.__class__).lower():
            # falcon-40b: 两组独立的 LN
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_attn,
                    layers=[module.self_attention.query_key_value],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_mlp,
                    layers=[module.mlp.dense_h_to_4h],
                    inp=input_feat["mlp.dense_h_to_4h"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        else:
            raise NotImplementedError(
                "Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported"
            )
        # MLP 输出: 激活函数 → dense_4h_to_h
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "bigcode" in str(module.__class__).lower():
        # -------------------- BigCode/SantaCoder 模型 --------------------
        # attention 输入: LN → fused c_attn
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat["attn.c_attn"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        # MLP 输入: LN → c_fc
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_2,
                layers=[module.mlp.c_fc],
                inp=input_feat["mlp.c_fc"],
                module2inspect=module.mlp,
            )
        )
        # MLP 输出: 激活函数 → c_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )
    elif "neox" in str(module.__class__).lower():
        # -------------------- GPT-NeoX 模型 --------------------
        # attention 输入: LN → fused QKV
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        # MLP 输入: LN → dense_h_to_4h
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
            )
        )
        # MLP 输出: 激活函数 → dense_4h_to_h
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif module.__class__.__name__ == "Qwen2DecoderLayer":
        # -------------------- Qwen2 模型 --------------------
        # 结构与 Llama 相同，但使用 Qwen2RMSNorm
        # attention 输入: LN → Q,K,V
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attention 输出: V → O（条件：形状一致）
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                )
            )
        # MLP 输入: LN → gate_proj, up_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
            )
        )
        # MLP 输出: up_proj → down_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
            )
        )
    elif module.__class__.__name__ == "InternLM2DecoderLayer":
        # -------------------- InternLM2 / InternVL2 模型 --------------------
        # 注意：InternLM2 使用 fused QKV（attention.wqkv），不是分开的 Q,K,V
        # attention 输入: attention_norm → wqkv (fused QKV)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attention_norm,
                layers=[
                    module.attention.wqkv,
                ],
                inp=input_feat["attention.wqkv"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        # attention 输出: wqkv → wo（条件：形状一致）
        if module.attention.wqkv.weight.shape == module.attention.wo.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.attention.wqkv,
                    layers=[module.attention.wo],
                    inp=input_feat["attention.wo"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                )
            )
        # MLP 输入: ffn_norm → w1, w3 (gate/up 投影)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn_norm,
                layers=[module.feed_forward.w1, module.feed_forward.w3],
                inp=input_feat["feed_forward.w1"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.feed_forward,
            )
        )
        # MLP 输出: w3 → w2 (up → down 投影)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.feed_forward.w3,
                layers=[module.feed_forward.w2],
                inp=input_feat["feed_forward.w2"],
                reweight_ratio=reweight_ratio_dict["mlp"],
            )
        )

    elif module.__class__.__name__ == "Qwen2VLDecoderLayer":  # 新加的Qwen2vl处理模块
        # -------------------- Qwen2-VL 模型 --------------------
        # 结构与 Qwen2 相同，MBQ 新增的多模态模型适配
        # attention 输入: LN → Q,K,V
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attention 输出: V → O（条件：形状一致）
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                )
            )
        # MLP 输入: LN → gate_proj, up_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
            )
        )
        # MLP 输出: up_proj → down_proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale(module, scales_list, input_feat_dict=None):
    """
    将搜索到的 scale 应用到模型权重上，并同步更新输入特征缓存。

    这是 scale 搜索的"落地"步骤——之前在网格搜索中只是临时施加 scale，
    现在将最优 scale 永久写入模型权重。

    参数:
        module:          完整模型（或父模块），用于 get_op_by_name 按名称查找子模块
        scales_list:     [(prev_op_name, (fc_names,), scales), ...]
        input_feat_dict: 输入激活值缓存 dict，如果提供则同步除以 scale
    """
    for prev_op_name, layer_names, scales in scales_list:
        # 通过名称在模型中找到实际的 nn.Module 对象
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        # 将相关模块移到 GPU 进行计算
        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        # ---- 根据 prev_op 类型分派具体的 scale 应用函数 ----
        if isinstance(prev_op, nn.Linear):
            # FC → FC 组：V → O, up → down 等
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        # 兼容多种 RMSNorm 实现：LlamaRMSNorm, InternLM2RMSNorm, Qwen2RMSNorm 以及标准 LayerNorm
        elif (
            isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm))
            or prev_op.__class__.__name__ == "InternLM2RMSNorm"
            or prev_op.__class__.__name__ == "Qwen2RMSNorm"
        ):
            # LN → FC 组：LN → QKV, LN → gate/up 等
            # LN 权重 /= scale, FC 权重 *= scale
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            # 激活函数 → FC：用 ScaledActivation 包装
            # 这是 Bloom / OPT 等模型的特殊处理
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # ---- 同步更新输入特征缓存 ----
        # 因为 LN 权重变了，下一层的输入激活值也需要除以 scale
        # 这样后续层的输入特征才能与修改后的权重保持一致
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        # 将模块移回 CPU，释放 GPU 内存
        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
