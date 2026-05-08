"""MBQ 量化搜索入口脚本。

整体流程与 README 中的说明保持一致：
1. 解析命令行参数或 YAML 配置；
2. 按 `model` 加载对应的多模态模型；
3. 根据 `calib_data` 构建文本或图文校准集；
4. 调用 `qwrapper` 执行量化搜索、伪量化或相关处理。
"""

import argparse
import datetime
import importlib
import json
import os
import sys
import traceback
import warnings
from functools import partial

import numpy as np
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from typing import Union

from lmms_eval.models import get_model

from qmllm.quantization.quant_wrapper import qwrapper
from qmllm.models import get_process_model
from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset


def parse_quant_args() -> argparse.Namespace:
    """解析量化搜索所需的命令行参数。

    该入口同时支持直接命令行传参和后续通过 `--config` 覆盖参数。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    # calibration parameters
    parser.add_argument("--calib_data", default="pileval", choices=["pileval", "coco", None])
    parser.add_argument("--n_samples", default=128, type=int)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)

    # TODO: quantization parameters
    parser.add_argument("--method", default="awq", choices=["awq", "smoothquant", "mbq", "rtn", None])
    parser.add_argument("--w_bit", default=8, type=int)
    parser.add_argument("--a_bit", default=16, type=int)
    parser.add_argument("--w_group", default=128, type=int)
    parser.add_argument("--alpha", default=0.5, type=int)
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--distort", action="store_true")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"])

    parser.add_argument("--scale_path", default=None, type=str)
    parser.add_argument("--run_process", action="store_true")
    parser.add_argument("--pseudo_quant", action="store_true")
    args = parser.parse_args()
    return args


def cli_quant(args: Union[argparse.Namespace, None] = None) -> None:
    """量化搜索的总入口。

    如果外部没有传入 `args`，就从命令行解析；如果指定了 YAML 配置，
    则会把配置文件中的每一组参数展开成一次独立的量化任务。
    """
    if not args:
        # 允许直接运行脚本时从命令行读取参数。
        args = parse_quant_args()

    args_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # YAML 里可以写单个配置，也可以写多个配置列表；这里统一展开成多个任务。
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    for args in args_list:
        cli_quant_single(args)


def cli_quant_single(args: Union[argparse.Namespace, None] = None) -> None:
    """执行一次完整的量化流程。

    这里先在 evaluator 外部完成模型加载和校准数据构建，
    方便在量化前拿到原始的多模态模型、tokenizer 和 processor。
    """
    if args.model_args is None:
        args.model_args = ""

    # 先根据 `model` 选择 lmms-eval 中对应的模型封装并加载基座模型。
    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )

    # 再根据具体模型类型构造预处理器，后续校准和量化都依赖这个包装后的模型。
    Process_ModelClass = get_process_model(args.model)
    process_model = Process_ModelClass(lm._model, 
                                       lm._tokenizer, 
                                       lm.processor if hasattr(lm, 'processor') else None)

    # 根据配置生成校准数据：pileval 走纯文本，coco 走多模态样本。
    prompt_inputs = None
    prompt_kwargs = None

    if args.calib_data == "pileval":
        prompt_inputs, prompt_kwargs = get_calib_dataset(data_path=args.data_path, tokenizer=lm._tokenizer, n_samples=args.n_samples)
    elif args.calib_data == "coco":
        prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(data_path=args.data_path,
                                                                    image_folder=args.image_folder,
                                                                    model=process_model,
                                                                    n_samples=args.n_samples,
                                                                    few_shot_format=args.few_shot_format,
                                                                    interleave_format=args.interleave_format,
                                                                    text_data_path=args.text_data_path)

    # 最后交给量化包装器，执行搜索、伪量化或结果保存等逻辑。
    qwrapper(process_model, prompt_inputs, prompt_kwargs, args)

    
if __name__ == "__main__":
    cli_quant()
