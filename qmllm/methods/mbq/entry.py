import os
import torch

from qmllm.methods.mbq.quantize.pre_quant import run_mbq, apply_mbq
from qmllm.methods.mbq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    pseudo_quantize_model_weight_act,
)


def mbq_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_mbq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: str = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 16,
    wa_quant: bool = False,
    reweight: bool = False,
    distort: bool = False,
    loss_mode: str = "mae",
    use_low_rank: bool = False,
    low_rank_rank: int = 16,
    low_rank_attn_topk_ratio: float = 0.4,
    low_rank_mlp_topk_ratio: float = 0.4,
    reweight_cache_path: str = None,
    linear_mixed_probe: bool = False,
    linear_probe_high_bit: int = 4,
    linear_probe_keep_ratio: float = 0.5,
):
    """
    model: here the model is the LLM, you have to extract the LLM first!
    prompt_tokens: the prompt tokens
    prompt_mask: the prompt mask, mask the answer language tokens
    run_mbq_process: whether to run the MBQ process
    """
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    assert scale_path is not None

    scale_exist = os.path.exists(scale_path)
    # reparameterization
    if run_mbq_process and not scale_exist:
        model.to_cpu()
        mbq_results = run_mbq(
            model,
            prompt_inputs,
            prompt_kwargs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            auto_scale=True,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            reweight=reweight,
            reweight_cache_path=reweight_cache_path,
            distort=distort,
            use_low_rank=use_low_rank,
            low_rank_rank=low_rank_rank,
            low_rank_attn_topk_ratio=low_rank_attn_topk_ratio,
            low_rank_mlp_topk_ratio=low_rank_mlp_topk_ratio,
            linear_mixed_probe=linear_mixed_probe,
            linear_probe_high_bit=linear_probe_high_bit,
            linear_probe_keep_ratio=linear_probe_keep_ratio,
        )

        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)

        torch.save(mbq_results, scale_path)
        print("MBQ results saved at", scale_path)
    else:
        print(
            f"MBQ scale file already exists at {scale_path}, skipping MBQ process and loading the scale directly."
        )

    if pseudo_quant:
        mbq_results = torch.load(scale_path, map_location="cpu")
        apply_mbq(model.model, mbq_results)

        if not wa_quant:
            # weight quantization
            pseudo_quantize_model_weight(
                model.model,
                w_bit=w_bit,
                q_config=q_config,
                low_rank_results=mbq_results.get("low_rank", []),
                linear_bit_map=mbq_results.get("linear_bit_map", {}),
            )
        else:
            # weight activation quantization
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()
    return model
