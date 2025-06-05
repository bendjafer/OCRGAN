#!/bin/bash

# ==== User-editable variables ====
PYTHON_EXEC="python3.7"
TRAIN_SCRIPT="train.py"
DATAROOT="data/processed/dagm_processed"
ISIZE=256
NITER=100
MODEL="ocr_gan_aug"
BATCHSIZE=64
GPU_ID=2

# Get current date in YYYYMMDD format
DATE=$(date +%Y%m%d)
HISTORY_FILE="output/history/training_history_${DATE}.log"

CLASSES=(
dagm_1
dagm_2
dagm_3
dagm_4
dagm_5
dagm_6
dagm_7
dagm_8
dagm_9
dagm_10
)

# Ensure the log directory exists
mkdir -p "$(dirname "$HISTORY_FILE")"

# Write history header if file doesn't exist
if [ ! -f "$HISTORY_FILE" ]; then
    echo -e "DATE\tCLASS\tMODEL\tBATCHSIZE\tISIZE\tNITER\tGPU_ID\tSTART_TIME\tEND_TIME\tDURATION_SEC\tTRAIN_SCRIPT\tDATAROOT\tRUN_NAME" > "$HISTORY_FILE"
fi

# ==== Training loop ====
for CLASS in "${CLASSES[@]}"; do
    CLASS_DATAROOT="$DATAROOT/$CLASS"
    NAME="${MODEL}_${CLASS}_${DATE}"

    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    START_SEC=$(date +%s)
    echo "=== Training on class: $CLASS === [$START_TIME]"

    $PYTHON_EXEC $TRAIN_SCRIPT \
      --dataset "$CLASS" \
      --dataroot "$CLASS_DATAROOT" \
      --isize "$ISIZE" \
      --niter "$NITER" \
      --model "$MODEL" \
      --batchsize "$BATCHSIZE" \
      --gpu_ids "$GPU_ID" \
      --name "$NAME" \
      --save_test_images

    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    END_SEC=$(date +%s)
    DURATION=$((END_SEC - START_SEC))

    # Save to history file
    echo -e "${DATE}\t${CLASS}\t${MODEL}\t${BATCHSIZE}\t${ISIZE}\t${NITER}\t${GPU_ID}\t${START_TIME}\t${END_TIME}\t${DURATION}\t${TRAIN_SCRIPT}\t${CLASS_DATAROOT}\t${NAME}" >> "$HISTORY_FILE"

    echo "=== Done: $CLASS === [$END_TIME] Duration: ${DURATION}s"
done