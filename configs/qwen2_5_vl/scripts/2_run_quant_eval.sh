#!/bin/bash
set -e
CONFIG_PATH="configs/qwen2_5_vl/Eval/my_3b_textvqa_val_eval.yaml"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/textvqa_val/qwen2_5_vl/qwen2_5_vl_3b_w2g32_scale_reweight_true_svd_1.0_mixed_0.3.log"
GPU_ID=3
mkdir -p "$(dirname "${LOG_FILE}")"
: > "${LOG_FILE}"
exec >> "${LOG_FILE}" 2>&1
echo "========================================="
echo "Eval started at: $(date)"
echo "========================================="
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "LOG_DIR: ${LOG_DIR}"
echo "LOG_FILE: ${LOG_FILE}"
echo "GPU_ID: ${GPU_ID}"
echo "========================================="
(
    trap '' HUP
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -W ignore main.py --config "${CONFIG_PATH}"
    exit_code=$?
    echo "========================================="
    if [ "${exit_code}" -eq 0 ]; then
        echo "eval successfully!"
    else
        echo "eval failed with exit code ${exit_code}!"
    fi
    echo "========================================="
    exit "${exit_code}"
) &
echo "========================================="
echo "Background job started"
echo "PID: $!"
echo "========================================="
