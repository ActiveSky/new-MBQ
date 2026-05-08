#!/bin/bash
# MBQ Quick PPL Evaluation Script
# 快速 PPL 评估脚本，用于检验量化效果

set -euo pipefail  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ========== 配置参数 ==========
MODEL_NAME="internvl2"
MODEL_PATH="OpenGVLab/InternVL2-8B"
DATASET="wikitext2"
N_SAMPLES=32
RESULT_DIR="$REPO_ROOT/outputs/ppl"

# ========== 创建目录 ==========
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "MBQ Quick PPL Evaluation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "=========================================="

# ========== Step 1: 评估原始模型 ==========
echo "Step 1: Evaluating FP16 model..."
python eval_ppl.py \
    --model $MODEL_NAME \
    --model_args pretrained=$MODEL_PATH \
    --dataset $DATASET \
    --n_samples $N_SAMPLES \
    --output_path "$RESULT_DIR/fp16_ppl.json" \
    --verbose



