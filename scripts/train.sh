#!/bin/bash
# Training script for LCCO-Net.
# Adjust DATA_PATH, OUTPUT_DIR and CHECKPOINT to your local paths.

# ---------------------------------------------------------------------------
# Paths (edit these before running)
# ---------------------------------------------------------------------------
DATA_PATH="/path/to/AIGCDetectBenchmark"     # training dataset root
OUTPUT_DIR="./output/lcco_resnet"            # logs and checkpoints
CHECKPOINT="./checkpoint/resnet50_aide.pth"  # AIDE-style ResNet pretrained weights
LOG_DIR="${OUTPUT_DIR}/logs"

# ---------------------------------------------------------------------------
# Model / loss configuration (paper defaults)
# ---------------------------------------------------------------------------
MODEL="LCCONet"
BACKBONE="resnet"          # resnet or convnext
ALPHA=1.0                  # LHFE high-frequency amplification
CSDD_WEIGHT=0.5            # lambda: weight of CSDD in L = CE + lambda*CSDD
FAD_BETA=0.5               # beta1: FAD weight inside CSDD
CCO_BETA=0.5               # beta2: CCO weight inside CSDD
MARGIN_COEFF=0.30          # coefficient of high-frequency energy in adaptive margin

# ---------------------------------------------------------------------------
# Training configuration (paper defaults)
# ---------------------------------------------------------------------------
BATCH_SIZE=32
EPOCHS=100
WEIGHT_DECAY=0.01
INPUT_SIZE=256
NUM_WORKERS=8

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_finetune.py \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --blr 5e-4 \
    --weight_decay ${WEIGHT_DECAY} \
    --input_size ${INPUT_SIZE} \
    --model ${MODEL} \
    --backbone ${BACKBONE} \
    --alpha ${ALPHA} \
    --csdd_weight ${CSDD_WEIGHT} \
    --fad_beta ${FAD_BETA} \
    --cco_beta ${CCO_BETA} \
    --margin_coeff ${MARGIN_COEFF} \
    --data_path ${DATA_PATH} \
    --nb_classes 2 \
    --num_workers ${NUM_WORKERS} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --resume ${CHECKPOINT} \
    --save_ckpt_freq 5 \
    --update_freq 1 \
    --use_amp
