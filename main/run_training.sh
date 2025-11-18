#!/usr/bin/env bash
set -e

###############################################
# USER CONFIGURATION
###############################################

# List of datasets
DATASETS=(
    "AmpHGT"
    "CellPPD"
    "MHC"
    "THPep"
)

# List of model names
MODELS=(
    "aaronfeller/PeptideMTR_sm"
    "aaronfeller/PeptideMTR_base"
    "aaronfeller/PeptideMTR_lg"
    "aaronfeller/PeptideMLM_sm"
    "aaronfeller/PeptideMLM_base"
    "aaronfeller/PeptideMLM_lg"
    "aaronfeller/PeptideMLM-MTR_sm"
    "aaronfeller/PeptideMLM-MTR_base"
    "aaronfeller/PeptideMLM-MTR_lg"
)

# List of GPU IDs available
GPUS=(0 1 2 3 4 5 6 7)

# Path to training script
TRAIN_SCRIPT="scripts/train_model.py"

# Output log directory
LOG_DIR="logs/launcher"
mkdir -p "$LOG_DIR"

###############################################
# JOB QUEUE LOGIC
###############################################

# Track PIDs per GPU
declare -A GPU_PIDS

function wait_for_free_gpu() {
    while true; do
        for GPU in "${GPUS[@]}"; do
            PID=${GPU_PIDS[$GPU]}

            # GPU is free if no process OR process finished
            if [[ -z "$PID" ]] || ! ps -p "$PID" > /dev/null 2>&1; then
                echo "$GPU"
                return
            fi
        done

        echo "[INFO] All GPUs busy — waiting 10 seconds..."
        sleep 10
    done
}

###############################################
# JOB LAUNCH
###############################################

JOB_ID=0

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        
        # if [[ "$MODEL" == *"_sm"* && "$DATASET" == "AmpHGT" ]]; then
        #     echo "[SKIP] Skipping already done: Dataset=$DATASET Model=$MODEL"
        #     continue
        # fi

        # if [[ "$MODEL" == *"_base"* && "$DATASET" == "AmpHGT" ]]; then
        #     echo "[SKIP] Skipping already done: Dataset=$DATASET Model=$MODEL"
        #     continue
        # fi

        GPU=$(wait_for_free_gpu)
        LOG_FILE="${LOG_DIR}/job_${JOB_ID}_${DATASET}/${MODEL//\//_}.log"

        mkdir -p "$(dirname "$LOG_FILE")"

        echo "[LAUNCH] Job $JOB_ID: Dataset=$DATASET Model=$MODEL --> GPU=$GPU"
        echo "  Log: $LOG_FILE"

        CUDA_VISIBLE_DEVICES=$GPU \
        python "$TRAIN_SCRIPT" \
            --dataset "$DATASET" \
            --gpu 0 \
            --model_name "$MODEL" \
            > "$LOG_FILE" 2>&1 &

        # Store PID for GPU tracking
        GPU_PIDS[$GPU]=$!

        JOB_ID=$((JOB_ID + 1))

        # short safety delay
        sleep 2
    done
done

echo "========================================="
echo "All jobs submitted. Monitoring queue..."
echo "========================================="

# Wait for all GPUs to finish
for GPU in "${GPUS[@]}"; do
    PID=${GPU_PIDS[$GPU]}
    if [[ -n "$PID" ]]; then
        echo "Waiting for GPU $GPU (PID $PID)..."
        wait "$PID"
        echo "GPU $GPU is done."
    fi
done

echo "========================================="
echo "All training runs completed."
echo "========================================="