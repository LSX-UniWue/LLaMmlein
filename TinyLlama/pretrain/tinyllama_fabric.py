import datetime
import functools
import glob
import io
import json
import math
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from lightning.fabric.plugins.precision import FSDPPrecision
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from lit_gpt import FusedCrossEntropyLoss
from lit_gpt.jsonl_dataset import CombinedDataset, JsonlDataset
from lit_gpt.model import (
    GPT,
    Block,
    CausalSelfAttention,
    Config,
    LLaMAMLP,
    xformersfree_LLaMAMLP,
)
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import (
    chunked_cross_entropy,
    get_default_supported_precision,
    num_parameters,
    step_csv_logger,
)
from pytorch_lightning.loggers import WandbLogger
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.utils.data import DataLoader
from transformers import set_seed

model_name = "LLaMmlein_7B"
name = os.environ.get("MODEL_NAME", "out/LLaMmlein_7B")

out_dir = Path("out") / name

PAD_ID = 32010

# Hyperparameters
num_devices_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 4))
num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
num_of_devices = num_devices_per_node * num_nodes
global_batch_size = 1024
learning_rate = 2.5e-4
micro_batch_size = min(8, global_batch_size // num_of_devices)
max_step = 1430512
warmup_steps = 2000
log_step_interval = 1000
eval_iters = 100
save_step_interval = 10000
eval_step_interval = max_step
log_datapoints = True

precision = get_default_supported_precision(training=True)
transformer_layer_cls = {
    Block,
}

activation_checkpointing_policy = {Block}
sharding_strategy = "SHARD_GRAD_OP"

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-6

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

assert global_batch_size % num_of_devices == 0
assert batch_size % gradient_accumulation_steps == 0
assert global_batch_size % gradient_accumulation_steps == 0
assert batch_size % micro_batch_size == 0
assert global_batch_size % micro_batch_size == 0

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

train_data_config = [
    ("train", 1.0),
]

val_data_config = [
    ("valid", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
wandb_logger = WandbLogger()


def setup(
    devices: int = 8,
    train_data_dir: Path = Path("dataset"),
    val_data_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
) -> None:
    tpu = False
    seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    if devices > 1:
        strategy = FSDPStrategy(
            process_group_backend="nccl",
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layer_cls,
            ),
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
            backward_prefetch=BackwardPrefetch.BACKWARD_POST,
            sharding_strategy=sharding_strategy,
            activation_checkpointing_policy=activation_checkpointing_policy,
            precision=FSDPPrecision(precision),
            timeout=datetime.timedelta(seconds=60*60*8),
        )

    else:
        strategy = "auto"

    fabric = L.Fabric(
        accelerator="cuda",
        devices=num_devices_per_node,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=[
            # logger,
            wandb_logger,
        ],
    )
    fabric.print(hparams)
    fabric.launch(main, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    if activation_checkpointing_policy is not None and config._mlp_class == "LLaMAMLP":
        config._mlp_class = "xformersfree_LLaMAMLP"

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )

    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    if resume:
        log_file_path = resume.with_name(resume.stem.replace("-ckpt", "")) / "{rank}.log"

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
        resume=log_file_path if resume else None,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    if log_datapoints:
        log_file_path = Path(f"/tmp/training_{fabric.global_rank}.log")
        assert not log_file_path.exists()
        log_file = open(log_file_path, "w", buffering=io.DEFAULT_BUFFER_SIZE * 1000)

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        del meta_model, x

    total_lengths = 0
    total_t0 = None
    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    initial_iter = state["iter_num"]

    loss_func = FusedCrossEntropyLoss()
    for train_data in train_dataloader:
        if total_t0 is None:
            total_t0 = time.perf_counter()
        if state["iter_num"] >= max_iters:
            break

        if log_datapoints:
            log_file.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "iter_num": state["iter_num"],
                        "data_id": train_data["data_id"],
                        "file_id": train_data["file_id"].cpu().numpy().tolist(),
                        "process_rank": list(set(train_data["process_rank"].cpu().numpy().tolist()))[0],
                    }
                )
                + "\n"
            )

        train_data = train_data["input_ids"]

        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()
        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, pad_id=PAD_ID)
            targets[targets == PAD_ID] = -100
            loss = loss_func(logits, targets)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        remaining_time = (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"]) / 3600
        fabric.print(
            f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {remaining_time:.2f} hours. "
            # print days as well
            f" or {remaining_time / 24:.2f} days. "
        )
        if state["iter_num"] % log_iter_interval == 0:

            fabric.log_dict(
                {
                    "lr": lr,
                    "remaining_hours": remaining_time,
                    "iter_time": (t1 - iter_t0) * 1000,
                    "iter_num": state["iter_num"],
                },
                state["step_count"],
            )

        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item(),
        )

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict(
                {
                    "metric/val_loss": val_loss.item(),
                    "metric/val_ppl": math.exp(val_loss.item()),
                    "total_tokens": model.config.block_size
                    * (state["iter_num"] + 1)
                    * micro_batch_size
                    * fabric.world_size,
                },
                state["step_count"],
            )
            fabric.barrier()
        if (not is_accumulating and state["step_count"] % save_step_interval == 0) or (
            state["iter_num"] >= max_iters - 1
        ):
            checkpoint_path = out_dir / f"iter-{state['iter_num']:08d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)
            if log_datapoints:
                log_file.flush()
                log_file.close()
                (out_dir / f"iter-{state['iter_num']:08d}").mkdir(parents=True, exist_ok=True)
                shutil.copy2(log_file_path, out_dir / f"iter-{state['iter_num']:08d}" / f"{fabric.global_rank}.log")
                log_file = open(log_file_path, "w", buffering=io.DEFAULT_BUFFER_SIZE * 1000)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        losses[k] = loss.item()

    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    num_workers: int = 1,
    pin_memory: bool = False,
    shuffle: bool = True,
    seed: int = 3407,
    split="train",
    resume: Optional[Path] = None,
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"*")))
        # fabric.print(filenames)
        random.seed(seed)
        random.shuffle(filenames)

        dataset = JsonlDataset(
            filenames,
            # n_chunks control the buffer size.
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            block_size=block_size,
            tokenizer_path=Path("../LLaMmlein_tok"),
            padding_id=PAD_ID,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
            resume=resume,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=100,
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 3407,
    resume: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
        resume=resume,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation",
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be True, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
