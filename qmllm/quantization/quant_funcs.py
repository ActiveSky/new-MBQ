import torch


def _normalize_double_quant_config(double_quant_config):
    if double_quant_config is None:
        double_quant_config = {}
    if not isinstance(double_quant_config, dict):
        raise TypeError("double_quant_config must be a dict.")

    normalized = {
        "scale_bit": 8,
        "scale_group": 128,
        "zero_point": True,
    }
    normalized.update(double_quant_config)
    normalized["scale_bit"] = int(normalized["scale_bit"])
    normalized["scale_group"] = int(normalized["scale_group"])
    normalized["zero_point"] = bool(normalized.get("zero_point", True))
    return normalized


@torch.no_grad()
def pseudo_quantize_scales(
    scales,
    n_bits=8,
    q_group_size=128,
    zero_point=True,
    eps=1e-12,
):
    """Pseudo-quantize primary quantization scales for double-quant simulation."""
    if n_bits >= 16:
        return scales

    org_shape = scales.shape
    flat = scales.reshape(-1)
    if flat.numel() <= 1:
        return scales

    q_group_size = int(q_group_size)
    if q_group_size <= 0:
        q_group_size = flat.numel()

    pad = (-flat.numel()) % q_group_size
    if pad:
        flat = torch.cat([flat, flat[-1:].expand(pad)], dim=0)

    grouped = flat.view(-1, q_group_size)
    qmax = 2**n_bits - 1

    if zero_point:
        min_val = grouped.amin(dim=1, keepdim=True)
        max_val = grouped.amax(dim=1, keepdim=True)
        dq_scale = (max_val - min_val).clamp(min=eps) / qmax
        q = torch.round((grouped - min_val) / dq_scale).clamp(0, qmax)
        dequant = q * dq_scale + min_val
    else:
        max_val = grouped.abs().amax(dim=1, keepdim=True).clamp(min=eps)
        dq_scale = max_val / qmax
        q = torch.round(grouped / dq_scale).clamp(0, qmax)
        dequant = q * dq_scale

    dequant = dequant.reshape(-1)
    if pad:
        dequant = dequant[:-pad]
    return dequant.reshape(org_shape)


@torch.no_grad()
def pseudo_quantize_tensor(
    tensor,
    n_bits=8,
    zero_point=True,
    q_group_size=-1,
    per_tensor=False,
    inplace=False,
    double_quant=False,
    double_quant_config=None,
):
    """
    Simulate quantization by quantizing and immediately dequantizing a tensor.

    When double_quant=True, the primary quantization scales are also
    pseudo-quantized before zero-points and quantized values are computed.
    This models the numerical effect of scale double quantization without
    changing the storage format.
    """
    org_tensor_shape = tensor.shape

    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    dq_config = None
    if double_quant:
        dq_config = _normalize_double_quant_config(double_quant_config)

    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        if dq_config is not None:
            scales = pseudo_quantize_scales(
                scales,
                n_bits=dq_config["scale_bit"],
                q_group_size=dq_config["scale_group"],
                zero_point=dq_config["zero_point"],
            )
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        if dq_config is not None:
            scales = pseudo_quantize_scales(
                scales,
                n_bits=dq_config["scale_bit"],
                q_group_size=dq_config["scale_group"],
                zero_point=dq_config["zero_point"],
            )
        zeros = 0

    if inplace:
        (
            (tensor.div_(scales).round_().add_(zeros))
            .clamp_(min_int, max_int)
            .sub_(zeros)
        ).mul_(scales)
    else:
        tensor = (
            torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int)
            - zeros
        ) * scales

    assert torch.isnan(tensor).sum() == 0
    return tensor.reshape(org_tensor_shape)


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, zero_point=False):
    return pseudo_quantize_tensor(
        w,
        n_bits=n_bits,
        zero_point=zero_point,
        q_group_size=-1,
        per_tensor=False,
        inplace=False,
    )


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(
        t,
        n_bits=n_bits,
        zero_point=zero_point,
        q_group_size=-1,
        per_tensor=False,
        inplace=False,
    )
    return t.reshape(t_shape)


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, zero_point=False):
    return pseudo_quantize_tensor(
        w,
        n_bits=n_bits,
        zero_point=zero_point,
        q_group_size=-1,
        per_tensor=True,
        inplace=False,
    )


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(
        t,
        n_bits=n_bits,
        zero_point=zero_point,
        q_group_size=-1,
        per_tensor=True,
        inplace=False,
    )
    return t.reshape(t_shape)
