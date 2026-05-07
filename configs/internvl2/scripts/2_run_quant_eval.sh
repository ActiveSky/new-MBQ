
#!/bin/bash

set -e



# Configuration
CONFIG_PATH="configs/internvl2/Eval/my_eval_ocrbench_smooth.yaml"  # Change this to your model path
LOG_DIR="./logs"  # Directory to save logs
LOG_FILE="${LOG_DIR}/8b_eval_out_ocrbench_w2g128_smooth.log"  # Log file name
GPU_ID=6  # GPU device ID

mkdir -p "${LOG_DIR}"
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


# run in background so the launcher returns immediately
(
    trap '' HUP
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -W ignore main.py \
        --config "${CONFIG_PATH}"
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


