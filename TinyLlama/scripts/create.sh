#!/bin/bash

#SBATCH --job-name="format"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:10:00

ITERATION_NUM="00034000"

OUT_DIR="out/LLaMmlein_7B"
CHECKPOINT="iter-$ITERATION_NUM-ckpt.pth"
MODEL_NAME="LLaMmlein_7B"
INFERENCE_DIR="models/LLaMmlein_7B"
TOKENIZER_PATH="LLaMmlein_tok"
FINAL_MODEL="models/hf/LLaMmlein_7B"
CONTAINER_PATH="llammlein.sif"
SCRIPT_PATH="scripts/convert_lit_checkpoint.py"

mkdir -p "$OUT_DIR"
mkdir -p "$INFERENCE_DIR"

apptainer exec $CONTAINER_PATH python3 $SCRIPT_PATH \
    --out_dir "$OUT_DIR" \
    --checkpoint_name "$CHECKPOINT" \
    --model_name "$MODEL_NAME" \
    --model_only False

if [ $? -eq 0 ]; then
    mv "$OUT_DIR/iter-$ITERATION_NUM-ckpt.bin" "$INFERENCE_DIR/pytorch_model.bin"
    mv "$OUT_DIR/config.json" "$INFERENCE_DIR/"
else
    echo "Conversion failed, skipping file moves."
fi

apptainer exec llammlein.sif python3 create.py --checkpoint "iter-$ITERATION_NUM-ckpt" --model_path $INFERENCE_DIR --tok_path $TOKENIZER_PATH --save_path $FINAL_MODEL