# 🐑 LLäMmlein Codebase

Welcome to the **LLäMmlein** codebase — used to train our German, decoder-only **LLäMmlein** model family from scratch.
We currently provide three model sizes — **120M**, **1B**, and **7B** — all trained on the **same curated dataset** for consistent scaling comparisons.
All components of the training process, including code, datasets, and intermediate checkpoints are **openly released**.
Find more information in our [paper](https://arxiv.org/pdf/2411.11171). 
This repository builds on [TinyLlama](https://github.com/jzhang38/TinyLlama) as its backbone.

### 🔍 New Highlight Features
* ⚡️ Flash Attention 3
* 🏎️ Fast dataloader with caching for improved speed
* ✏️ Datapoint logging for better tracking



## Model Family

All models are available on Hugging Face, along with:
- Intermediate checkpoints
- Data logging metadata

| Model Size | Hugging Face Link |
|------------|-------------------|
| **LLäMmlein 7B**   | [LLäMmlein_7B](https://huggingface.co/LSX-UniWue/LLaMmlein_7B) |
| **LLäMmlein 1B**   | [LLäMmlein_1B](https://huggingface.co/LSX-UniWue/LLaMmlein_1B) |
| **LLäMmlein 120M** | [LLäMmlein_120M](https://huggingface.co/LSX-UniWue/LLaMmlein_120M) |

### Legacy Models (Preregistered, No Data Logging)

These earlier versions were used in our initial experiments (see accompanying paper):

| Model | Hugging Face Link |
|-------|-------------------|
| **LLäMmlein 1B (prerelease)**   | [LLäMmlein_1B_prerelease](https://huggingface.co/LSX-UniWue/LLaMmlein_1B_prerelease) |
| **LLäMmlein 120M (prerelease)** | [LLäMmlein_120M_prerelease](https://huggingface.co/LSX-UniWue/LLaMmlein_120M_prerelease) |


## Data 
All our models are trained using our filtered version of the [RedPajama V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) dataset, available under [LLaMmlein-Dataset](https://huggingface.co/datasets/LSX-UniWue/LLaMmlein-Dataset)
In addition to **paragraph deduplication**, we applied a **token-to-word ratio filter** to exclude low-quality or noisy text.

## Usage

### Install Requirements 🚀

We provide our requirements in `TinyLlama/cluster/building/requirements.txt`. They can be installed using: 
```bash
pip install -r requirements.txt
```

In addition, we provide a containerized environment using **Singularity (Apptainer)**.
The `.sif` image can be build using the provided definition file `llammlein.def`.
(located in `TinyLlama/cluster/building/`):

```bash 
bash build-image.sh
```

### Training 

The `tinyllama_fabric.py` contains the general training script. Before starting several parameters have to be set, we will list the most important here:  
* model_name : Identifier specified in lit-gpt/config.py
* global_batch_size: batch size across all gpus 
* learning_rate
* micro_batch_size
* max_steps 
* sharding_strategy: Efficiency of the FSDP sharding strategy can differ drastically between cluster settings
* train_data_dir: Path to the training dataset (line 112)
* state_dict_type: Depending on the size of model you are training and available gpus it can make sense to set this parameter to `sharded` instead of `full` (line 131)
* tokenizer_path: Path to the tokenizer (line 417)

Once all training parameters are configured, training can be launched using a script similar to this one: 
```bash
#!/bin/bash

#SBATCH --job-name="training"
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --time=24:00:00

# Set all cluster specific variables i.e. HTTP_PROXY, NCCL_IB_HCA ...

export MODEL_NAME="$OUTDIR" # location to save intermediate checkpoints
srun apptainer exec llammlein.sif python pretrain/tinyllama_fabric.py 
```

Our specific `.sh` file with all our selected environment variables can be found in `TinyLlama/cluster/exec`.
Training can be resumed from a specific checkpoint using:  
```bash
srun apptainer exec llammlein.sif python pretrain/tinyllama_fabric.py --resume $MODEL_NAME/iter-00100000-ckpt.pth
```

### Transformation 
The checkpoints saved during training are not Huggingface compatible therefore the examplary script `scripts/create.sh` has to be executed. 
Please make sure to add all paths correctly in the `.sh file`. 


---
### License
We release the LLäMmlein models under a research-only RAIL-M license. See [license.md](./license.md) for details.

### Citation
