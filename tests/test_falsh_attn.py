import torch


def _get_flash_attn_func():
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        try:
            from flash_attn.flash_attn_interface import flash_attn_func
        except ImportError as exc:
            raise AssertionError("flash_attn is not installed or does not expose flash_attn_func.") from exc

    return flash_attn_func


def _run_minimal_flash_attn(dtype: torch.dtype) -> None:
    flash_attn_func = _get_flash_attn_func()

    query = torch.randn(1, 4, 2, 16, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    output = flash_attn_func(query, key, value, 0.0, causal=True)
    torch.cuda.synchronize()

    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert output.is_cuda


def test_flash_attn_available():
    assert torch.cuda.is_available(), "CUDA is not available. flash-attn requires a CUDA-capable GPU."

    last_error = None
    for dtype in (torch.float16, torch.bfloat16):
        try:
            _run_minimal_flash_attn(dtype)
            return
        except Exception as exc:
            last_error = exc

    raise AssertionError(f"flash-attn import succeeded, but a minimal forward pass failed: {last_error}") from last_error


if __name__ == "__main__":
    test_flash_attn_available()
    print("flash-attn is available and passed a minimal CUDA forward-pass test.")
