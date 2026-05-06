import torch

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False):
    """
    模拟量化函数：对张量执行 量化 → 反量化 操作，模拟量化引入的数值误差。
    用于 weight / activation / KV cache 的量化感知 scale 搜索。

    参数:
        tensor:      输入张量，形状任意（内部会 reshape 为 2D）
        n_bits:      量化位宽，如 4/8/16
        zero_point:  是否使用非对称量化（含零点偏移），True=非对称，False=对称
        q_group_size:分组量化的组大小，-1 表示不分组（per-channel）
        per_tensor:  是否采用 per-tensor 量化（仅一个 scale/zero），默认 per-channel
        inplace:     是否原地修改输入张量

    返回:
        量化-反量化后的张量（模拟精度损失后的结果）
    """
    # 保存原始形状，后续恢复
    org_tensor_shape = tensor.shape

    # ---- 1. reshape 为 2D，便于 per-channel / per-group 量化 ----
    if q_group_size > 0:
        # 分组量化：[..., hidden] → [-1, q_group_size]
        # 每组独立计算 scale 和 zero_point
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        # per-tensor 量化：整张量共用一个 scale/zero_point
        # 全部行展平为一行 [1, -1]
        tensor = tensor.reshape(1, -1)
    # 确保是 2D 张量：[channels, features_per_channel]
    assert tensor.dim() == 2

    # ---- 2. 计算量化参数 (scale 和 zero_point) ----
    if zero_point:
        # 非对称量化（有零点偏移）：
        #   量化:   q = round((x - min) / scale)    范围 [0, 2^n - 1]
        #   反量化: x̂ = (q - zero_point) * scale
        max_val = tensor.amax(dim=1, keepdim=True)   # 每行最大值 [C, 1]
        min_val = tensor.amin(dim=1, keepdim=True)   # 每行最小值 [C, 1]
        max_int = 2**n_bits - 1                       # 量化上限（无符号整数最大值）
        min_int = 0                                   # 量化下限
        scales = (max_val - min_val).clamp(min=1e-5) / max_int  # scale = (max - min) / (2^n - 1)
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)  # zero_point = -round(min/scale)
    else:
        # 对称量化（无零点偏移）：
        #   量化:   q = round(x / scale)            范围 [-(2^{n-1}-1), 2^{n-1}-1]
        #   反量化: x̂ = q * scale
        max_val = tensor.abs().amax(dim=1, keepdim=True)  # 每行绝对值最大值 [C, 1]
        max_val = max_val.clamp(min=1e-5)                  # 防止除零
        max_int = 2 ** (n_bits - 1) - 1                    # 量化上限（有符号正半轴）
        min_int = -(2 ** (n_bits - 1))                     # 量化下限（有符号负半轴）
        scales = max_val / max_int                         # scale = |max| / (2^{n-1} - 1)
        zeros = 0                                          # 对称量化 zero_point = 0

    # ---- 3. 执行量化 → 反量化（模拟精度损失） ----
    if inplace:
        # 原地模式：直接修改输入张量
        #   x / scale → round → +zero → clamp → -zero → *scale
        (
            (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        # 非原地模式：生成新张量，不修改输入
        #   round(x / scale) + zero → clamp → -zero → *scale
        tensor = (
            torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
        ) * scales

    # 检查是否有 NaN（数值不稳定）
    assert torch.isnan(tensor).sum() == 0

    # ---- 4. 恢复原始形状 ----
    tensor = tensor.reshape(org_tensor_shape)

    return tensor


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, zero_point=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return t.reshape(t_shape)
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, zero_point=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape)
