import argparse
import torch
from transformers import AutoModelForCausalLM
import os
import shutil
import random
import numpy as np

def set_seeds():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

set_seeds()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default="iter-00200000-ckpt", help='Checkpoint name (without iter prefix and extension)')
parser.add_argument('--model_path', type=str, default="models/LLaMmlein_7B", help='Checkpoint name (without iter prefix and extension)')
parser.add_argument('--tok_path', type=str, default="LLaMmlein_tok", help='Checkpoint name (without iter prefix and extension)')
parser.add_argument('--save_path', type=str, default="models/hf/LLaMmlein_7B", help='Checkpoint name (without iter prefix and extension)')

args = parser.parse_args()

checkpoint = args.checkpoint
model_path = args.model_path
tokenizer_path = args.tok_path
new_model_path = args.save_path

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_path, weights_only=False)

print(f"Saving model to {new_model_path}")
model.save_pretrained(new_model_path, safe_serialization=False)

print("Copying tokenizer files...")
tokenizer_files = ["tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
for file_name in tokenizer_files:
    src = os.path.join(tokenizer_path, file_name)
    dst = os.path.join(new_model_path, file_name)
    shutil.copy(src, dst)

print("Model and tokenizer files saved successfully.")