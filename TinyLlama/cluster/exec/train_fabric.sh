#!/bin/bash

#SBATCH --job-name="training"
#SBATCH --output=outfiles/out_%j.txt
#SBATCH --error=outfiles/err_%j.txt
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --mail-user=

export HTTP_PROXY="http_proxy"
export HTTPS_PROXY="https_proxy"

export WANDB_API_KEY="key"
export WANDB_PROJECT="LLaMmlein"
export WANDB_ENTITY="entry"

export NCCL_DEBUG=WARN
export NCCL_PROTO=simple

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn


export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_HCA="=mlx5_2,mlx5_3,mlx5_4,mlx5_5"
export TRITON_CACHE_DIR="/tmp"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345

export MODEL_NAME="out/LLaMmlein_7B"
export PYTHONPATH=$PYTHONPATH:/projects/llammlein/TinyLlama/

srun apptainer exec --bind workspace llammlein.sif python pretrain/tinyllama_fabric.py