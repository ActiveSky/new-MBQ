"""
MBQ PPL (Perplexity) Evaluation Script

快速检验 MBQ 量化效果的评估脚本，支持：
1. WikiText2、PTB、C4等标准语言建模数据集
2. 加载 MBQ 量化结果（scale_path）
3. MBQ 伪量化前后对比
"""

import argparse
import json
import math
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

warnings.simplefilter("ignore", category=DeprecationWarning)

from lmms_eval.models import get_model
from qmllm.models import get_process_model
from qmllm.methods.mbq.quantize.pre_quant import apply_mbq
from qmllm.methods.mbq.quantize.quantizer import (
    pseudo_quantize_model_weight as mbq_pseudo_quantize_model_weight,
    pseudo_quantize_model_weight_act as mbq_pseudo_quantize_model_weight_act,
)


def parse_ppl_args():
    """解析 PPL 评估参数"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Model arguments
    parser.add_argument(
        "--model", default="hf", help="Model type: hf, internvl2, llava_onevision, etc."
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="Model arguments, e.g. pretrained=OpenGVLab/InternVL2-8B",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda, cuda:0, cpu"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "pileval", "all"],
        help="Dataset for PPL evaluation",
    )
    parser.add_argument(
        "--n_samples", type=int, default=128, help="Number of samples for evaluation"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048, help="Sequence length for truncation"
    )
    parser.add_argument("--data_path", type=str, default="", help="Custom dataset path")

    # Quantization arguments
    parser.add_argument(
        "--scale_path",
        type=str,
        default=None,
        help="Path to serialized MBQ state saved by torch.save",
    )
    parser.add_argument("--w_bit", type=int, default=4, help="Weight quantization bits")
    parser.add_argument(
        "--a_bit", type=int, default=16, help="Activation quantization bits"
    )
    parser.add_argument(
        "--w_group", type=int, default=128, help="Weight quantization group size"
    )
    parser.add_argument(
        "--zero_point",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use asymmetric quantization with zero point",
    )
    parser.add_argument(
        "--pseudo_quant",
        action="store_true",
        help="Apply MBQ pseudo quantization before PPL evaluation",
    )

    # Output arguments
    parser.add_argument(
        "--output_path", type=str, default="ppl_results.json", help="Output JSON path"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")

    args = parser.parse_args()
    return args


class PPLDataset:
    """PPL 评估数据集加载器"""

    @staticmethod
    def load_wikitext2(tokenizer, n_samples=128, seq_len=2048):
        """加载 WikiText2 数据集"""
        logger.info("Loading WikiText2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return PPLDataset._process_text_dataset(
            dataset, tokenizer, n_samples, seq_len, "text"
        )

    @staticmethod
    def load_ptb(tokenizer, n_samples=128, seq_len=2048):
        """加载 PTB 数据集"""
        logger.info("Loading PTB dataset...")
        dataset = load_dataset("ptb_text_only", "penn_treebank", split="test")
        return PPLDataset._process_text_dataset(
            dataset, tokenizer, n_samples, seq_len, "sentence"
        )

    @staticmethod
    def load_c4(tokenizer, n_samples=128, seq_len=2048):
        """加载 C4 数据集"""
        logger.info("Loading C4 dataset...")
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        return PPLDataset._process_streaming_dataset(
            dataset, tokenizer, n_samples, seq_len, "text"
        )

    @staticmethod
    def load_pileval(tokenizer, n_samples=128, seq_len=2048, data_path=""):
        """加载 Pile-Val 数据集"""
        logger.info("Loading Pile-Val dataset...")
        if data_path:
            dataset = load_dataset(data_path, split="validation")
        else:
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        return PPLDataset._process_text_dataset(
            dataset, tokenizer, n_samples, seq_len, "text"
        )

    @staticmethod
    def _process_text_dataset(dataset, tokenizer, n_samples, seq_len, text_field):
        """处理文本数据集"""
        samples = []
        n_run = 0

        for data in tqdm(dataset, desc="Processing dataset"):
            text = data[text_field].strip()
            if len(text) < 50:  # 跳过过短的样本
                continue

            encoded = tokenizer.encode(text, truncation=True, max_length=seq_len)
            if len(encoded) < 10:  # 跳过编码后过短的样本
                continue

            samples.append(torch.tensor([encoded]))
            n_run += 1
            if n_run >= n_samples:
                break

        logger.info(f"Loaded {len(samples)} samples")
        return samples

    @staticmethod
    def _process_streaming_dataset(dataset, tokenizer, n_samples, seq_len, text_field):
        """处理流式数据集"""
        samples = []
        n_run = 0

        for data in tqdm(
            dataset.take(n_samples * 10), desc="Processing streaming dataset"
        ):
            text = data[text_field].strip()
            if len(text) < 50:
                continue

            encoded = tokenizer.encode(text, truncation=True, max_length=seq_len)
            if len(encoded) < 10:
                continue

            samples.append(torch.tensor([encoded]))
            n_run += 1
            if n_run >= n_samples:
                break

        logger.info(f"Loaded {len(samples)} samples")
        return samples


@torch.no_grad()
def compute_perplexity(
    model, samples: List[torch.Tensor], device="cuda", verbose=False
):
    """
    计算模型的困惑度 (Perplexity)

    Args:
        model: 语言模型
        samples: 输入样本列表
        device: 计算设备
        verbose: 是否打印详细信息

    Returns:
        ppl: 困惑度值
        nll: 平均负对数似然
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for sample in tqdm(samples, desc="Computing PPL", disable=not verbose):
        input_ids = sample.to(device)
        labels = input_ids.clone()

        # 前向传播
        outputs = model(input_ids=input_ids, labels=labels)

        # 计算负对数似然
        nll = outputs.loss.item()
        n_tokens = input_ids.numel()

        total_nll += nll * n_tokens
        total_tokens += n_tokens

    # 计算困惑度: PPL = exp(NLL / N_tokens)
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    return ppl, avg_nll


def load_mbq_state(scale_path: Optional[str]):
    """加载 MBQ 量化结果。"""
    if not scale_path:
        raise ValueError("scale_path is required when pseudo_quant is enabled")

    if not os.path.exists(scale_path):
        raise FileNotFoundError(f"Scale file not found: {scale_path}")

    logger.info(f"Loading MBQ state from {scale_path}")
    return torch.load(scale_path, map_location="cpu")


def apply_mbq_quantization_to_model(model, quant_state, args):
    """将 MBQ 量化状态应用到 process_model.model。"""
    base_model = model.model
    q_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.w_group,
    }
    wa_quant = args.w_bit < 16 and args.a_bit < 16

    if quant_state is None:
        raise ValueError("MBQ evaluation requires a saved scale_path")

    apply_mbq(base_model, quant_state)
    if wa_quant:
        mbq_pseudo_quantize_model_weight_act(
            base_model, w_bit=args.w_bit, a_bit=args.a_bit
        )
    else:
        mbq_pseudo_quantize_model_weight(
            base_model, w_bit=args.w_bit, q_config=q_config
        )

    return model


def evaluate_single_dataset(model, tokenizer, dataset_name: str, args) -> Dict:
    """评估单个数据集的 PPL"""
    logger.info(f"Evaluating on {dataset_name}...")

    # 加载数据集
    if dataset_name == "wikitext2":
        samples = PPLDataset.load_wikitext2(tokenizer, args.n_samples, args.seq_len)
    elif dataset_name == "ptb":
        samples = PPLDataset.load_ptb(tokenizer, args.n_samples, args.seq_len)
    elif dataset_name == "c4":
        samples = PPLDataset.load_c4(tokenizer, args.n_samples, args.seq_len)
    elif dataset_name == "pileval":
        samples = PPLDataset.load_pileval(
            tokenizer, args.n_samples, args.seq_len, args.data_path
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 计算 PPL
    ppl, nll = compute_perplexity(model, samples, args.device, args.verbose)

    results = {
        "dataset": dataset_name,
        "ppl": ppl,
        "nll": nll,
        "n_samples": len(samples),
        "seq_len": args.seq_len,
        "w_bit": args.w_bit,
        "a_bit": args.a_bit,
        "quantized": args.pseudo_quant,
        "method": "mbq" if args.pseudo_quant else "fp16",
    }

    logger.info(f"{dataset_name} Results: PPL={ppl:.4f}, NLL={nll:.4f}")
    return results


def main():
    """PPL 评估主流程"""
    args = parse_ppl_args()

    # 设置日志
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    else:
        logger.remove()
        logger.add(sys.stdout, level="WARNING")

    logger.info("=" * 60)
    logger.info("MBQ Perplexity Evaluation")
    logger.info("=" * 60)

    # ========== Step 1: 加载模型 ==========
    logger.info("Loading model...")
    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )

    # 获取内部模型和 tokenizer
    model = lm._model
    tokenizer = lm._tokenizer

    # 构造处理模型（用于量化）
    Process_ModelClass = get_process_model(args.model)
    process_model = Process_ModelClass(
        model, tokenizer, lm.processor if hasattr(lm, "processor") else None
    )

    logger.info(f"Model loaded: {args.model_args}")

    # ========== Step 2: 应用 MBQ 量化（如果启用） ==========
    if args.pseudo_quant:
        logger.info("Applying MBQ quantization...")
        quant_state = load_mbq_state(args.scale_path)
        process_model = apply_mbq_quantization_to_model(
            process_model, quant_state, args
        )
        logger.info(
            f"MBQ quantization applied: w_bit={args.w_bit}, a_bit={args.a_bit}, pseudo_quant={args.pseudo_quant}"
        )

    # ========== Step 3: 评估 PPL ==========
    results_list = []

    if args.dataset == "all":
        # 评估所有数据集
        for dataset_name in ["wikitext2", "ptb", "c4", "pileval"]:
            try:
                result = evaluate_single_dataset(
                    process_model, tokenizer, dataset_name, args
                )
                results_list.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {dataset_name}: {e}")
                continue
    else:
        # 评估单个数据集
        result = evaluate_single_dataset(process_model, tokenizer, args.dataset, args)
        results_list.append(result)

    if not results_list:
        raise RuntimeError("No valid PPL results were produced")

    # ========== Step 4: 保存结果 ==========
    logger.info(f"Saving results to {args.output_path}")

    output_data = {
        "args": vars(args),
        "results": results_list,
        "summary": {
            "avg_ppl": np.mean([r["ppl"] for r in results_list]),
            "avg_nll": np.mean([r["nll"] for r in results_list]),
            "quantized": args.pseudo_quant,
            "method": "mbq" if args.pseudo_quant else "fp16",
        },
    }

    with open(args.output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ========== Step 5: 打印结果摘要 ==========
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)

    for result in results_list:
        logger.info(
            f"{result['dataset']}: PPL={result['ppl']:.4f} | "
            f"NLL={result['nll']:.4f} | "
            f"Method={result['method']}"
        )

    logger.info(f"Average PPL: {output_data['summary']['avg_ppl']:.4f}")
    logger.info(f"Average NLL: {output_data['summary']['avg_nll']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
