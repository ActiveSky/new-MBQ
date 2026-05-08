#!/bin/bash
# MBQ Quick PPL Evaluation Script
# 快速 PPL 评估脚本，用于检验量化效果

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ========== 配置参数 ==========
GPU_ID=0  # GPU device ID
MODEL_NAME="internvl2"
MODEL_PATH="OpenGVLab/InternVL2-8B"
DATASET="wikitext2"
N_SAMPLES=256
RESULT_DIR="$REPO_ROOT/outputs/ppl"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULT_DIR/eval_fp16_${RUN_TS}.log"
RESULT_JSON="$RESULT_DIR/fp16_${RUN_TS}_ppl.json"

# ========== 创建目录 ==========
mkdir -p "$RESULT_DIR"
: > "$LOG_FILE"
exec >> "$LOG_FILE" 2>&1

# ========== Step 1: 评估原始模型 ==========
(
    trap '' HUP
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python tests/test_PPL/eval_ppl.py \
        --model "$MODEL_NAME" \
        --model_args "pretrained=$MODEL_PATH" \
        --dataset "$DATASET" \
        --n_samples "$N_SAMPLES" \
        --output_path "$RESULT_JSON" \
        --verbose
    exit_code=$?
    echo "========================================="
    if [ "${exit_code}" -eq 0 ]; then
        echo "FP16 PPL evaluation completed successfully!"
    else
        echo "FP16 PPL evaluation failed with exit code ${exit_code}!"
    fi
    echo "========================================="
    exit "${exit_code}"
) &

echo "========================================="
echo "Background job started, PID: $!"
echo "Log: ${LOG_FILE}"
echo "========================================="
