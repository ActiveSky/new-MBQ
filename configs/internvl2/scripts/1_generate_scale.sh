#!/bin/bash

set -e



# Configuration
CONFIG_PATH="configs/internvl2/MBQ_search/my_8b_weight_only_smooth.yaml"  # Change this to your model path
LOG_DIR="./logs"  # Directory to save logs
LOG_FILE="${LOG_DIR}/8b_out_w2g64_smooth.log"  # Log file name
GPU_ID=6  # GPU device ID

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"
exec >> "${LOG_FILE}" 2>&1

echo "========================================="
echo "Generating  Scales"
echo "========================================="
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "LOG_DIR: ${LOG_DIR}"
echo "LOG_FILE: ${LOG_FILE}"
echo "GPU_ID: ${GPU_ID}"
echo "========================================="


# run in background so the launcher returns immediately
(
    trap '' HUP
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -W ignore main_quant.py \
        --config "${CONFIG_PATH}"
    exit_code=$?
    echo "========================================="
    if [ "${exit_code}" -eq 0 ]; then
        echo "Activation scales generated successfully!"
    else
        echo "Activation scales generation failed with exit code ${exit_code}!"
    fi
    echo "========================================="
    exit "${exit_code}"
) &

echo "========================================="
echo "Background job started"
echo "PID: $!"
echo "========================================="
