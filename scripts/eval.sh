#!/bin/bash
# Evaluation script for LCCO-Net.
# Iterates over all generator sub-directories under EVAL_ROOT and reports
# per-generator accuracy / AP / realacc / fakeacc and the mean accuracy.

# ---------------------------------------------------------------------------
# Paths (edit these before running)
# ---------------------------------------------------------------------------
EVAL_ROOT="/path/to/AIGCDetectBenchmark"   # benchmark root containing generator folders
OUTPUT_DIR="./output/lcco_resnet"          # where the result CSV is written
CHECKPOINT="./checkpoint/lcco_resnet_best.pth"  # trained LCCO-Net checkpoint

# ---------------------------------------------------------------------------
# Model / loss configuration (must match training)
# ---------------------------------------------------------------------------
MODEL="LCCONet"
BACKBONE="resnet"          # resnet or convnext
ALPHA=1.0
CSDD_WEIGHT=0.5
FAD_BETA=0.5
CCO_BETA=0.5
MARGIN_COEFF=0.30

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
BATCH_SIZE=32
INPUT_SIZE=256
NUM_WORKERS=8

mkdir -p "${OUTPUT_DIR}"

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 main_finetune.py \
    --eval \
    --batch_size ${BATCH_SIZE} \
    --input_size ${INPUT_SIZE} \
    --model ${MODEL} \
    --backbone ${BACKBONE} \
    --alpha ${ALPHA} \
    --csdd_weight ${CSDD_WEIGHT} \
    --fad_beta ${FAD_BETA} \
    --cco_beta ${CCO_BETA} \
    --margin_coeff ${MARGIN_COEFF} \
    --eval_data_path ${EVAL_ROOT} \
    --nb_classes 2 \
    --num_workers ${NUM_WORKERS} \
    --output_dir ${OUTPUT_DIR} \
    --resume ${CHECKPOINT} \
    --use_amp
