"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from xformers.ops import SwiGLU

# Depends on your root folder
try:
    from lit_gpt.config import Config
except ImportError:
    from TinyLlama.lit_gpt.config import Config

from .fused_rotary_embedding import apply_rotary_emb_func

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
# Don't agree with the versioning especially since padding received major updates after a while...
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")
# Beta version check - versioning doesn't really exist so make sure that you properly install from source
FlashAttention3Available = RequirementCache("flashattn-hopper>=3.0.0.b1")
FlashAttentionAvailable = FlashAttention2Available or FlashAttention3Available

# Check for environment variable to override FlashAttention availability
if "FLASH_ATTENTION_VERSION" in os.environ:
    flash_attention_env = os.getenv("FLASH_ATTENTION_VERSION", "0").lower()

    if flash_attention_env == "2":
        FlashAttentionAvailable = FlashAttention2Available = True
        FlashAttention3Available = False
    elif flash_attention_env == "3":
        FlashAttentionAvailable = FlashAttention3Available = True
        FlashAttention2Available = False
    elif flash_attention_env == "0":
        FlashAttentionAvailable = FlashAttention2Available = FlashAttention3Available = False
    else:
        raise ValueError(
            f"Invalid value for FLASH_ATTENTION_VERSION: {flash_attention_env}."
            " Please use 0 to disable, 2 or 3 to force FlashAttention 2 or 3 respectively."
        )


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():  # FIXME nach swiglu removal muss das evtl auch fixed werden
            if (name == "proj.weight" and isinstance(module, xformersfree_LLaMAMLP)) or (
                    name == "w3.weight"
                    and isinstance(module, SwiGLU)
                    or (name == "proj.weight" and isinstance(module, CausalSelfAttention))
            ):  # if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def forward(
            self,
            idx: torch.Tensor,
            max_seq_length: Optional[int] = None,
            input_pos: Optional[torch.Tensor] = None,
            pad_id: Optional[int] = None,
    ) -> torch.Tensor:
        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                    max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        cos, sin = self.rope_cache
        if use_kv_cache:
            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        fa_meta_information = None
        # print(f"{FlashAttentionAvailable=}, {bool(FlashAttentionAvailable)=}")
        # print(f"{x.device.type=}")
        # print(f"{x.dtype=}")
        # print(f"{mask=}")

        fa_is_available = (
                FlashAttentionAvailable
                # FA only works on GPUs with half precision
                and x.device.type == "cuda"
                # and x.dtype in (torch.float16, torch.bfloat16) # FIXME should be recalculated before each forward pass, but ignoring for now
                # seems to be related to inference
                and mask is None
        )
        # print(f"{fa_is_available=}")
        if (
                fa_is_available
                # needed to extract cu_seqs etc - **IMPORTANT** pad_id needs to be unique
                and pad_id is not None
        ):
            from .fa_padding import upad_meta_information

            fa_meta_information = upad_meta_information(input_ids=idx, pad_id=pad_id)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(
                    x,
                    (cos, sin),
                    max_seq_length,
                    fa_is_available=fa_is_available,
                    fa_meta_information=fa_meta_information,
                )
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, cos.size(-1) * 2)
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x,
                    (cos, sin),
                    max_seq_length,
                    mask,
                    input_pos,
                    self.kv_caches[i],
                    fa_is_available=fa_is_available,
                    fa_meta_information=fa_meta_information,
                )

        x = self.transformer.ln_f(x)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (
            B,
            max_seq_length,
            heads,
            rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
        )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config

    def forward(
            self,
            x: torch.Tensor,
            rope: RoPECache,
            max_seq_length: int,
            mask: Optional[torch.Tensor] = None,
            input_pos: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
            fa_is_available: Optional[bool] = None,
            fa_meta_information: Optional[Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(
            n_1, rope, max_seq_length, mask, input_pos, kv_cache, fa_is_available, fa_meta_information
        )
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )

            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


import inspect

# TODO possibly add a flag to enable flexible switching - for now greedy selection on what's available
if FlashAttention3Available:
    print("Using FlashAttention 3")
    from flash_attn_interface import flash_attn_func as _flash_attn_func
    from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func

    print(inspect.getfile(_flash_attn_func))
    flash_attn_func = lambda *args, **kwargs: _flash_attn_func(*args, **kwargs)[0]
    flash_attn_varlen_func = lambda *args, **kwargs: _flash_attn_varlen_func(*args, **kwargs)[0]


else:
    print("Using FlashAttention 2")
    from flash_attn.flash_attn_interface import flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func

    flash_attn_varlen_func = lambda *args, **kwargs: (
        kwargs.pop('seqused_q', None),
        kwargs.pop('seqused_k', None),
        _flash_attn_varlen_func(*args, **kwargs)
    )

    print(inspect.getfile(flash_attn_func))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
            self,
            x: torch.Tensor,
            rope: RoPECache,
            max_seq_length: int,
            mask: Optional[torch.Tensor] = None,
            input_pos: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
            fa_is_available: Optional[bool] = None,
            fa_meta_information: Optional[Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )  # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.config.head_size)
        v = v.reshape(B, T, -1, self.config.head_size)

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        # n_elem = int(self.config.rotary_percentage * self.config.head_size)

        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(
            q, k, v, mask=mask, fa_is_available=fa_is_available, fa_meta_information=fa_meta_information
        )

        y = y.contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            fa_is_available: Optional[bool] = None,
            fa_meta_information: Optional[Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]] = None,
    ):
        batch, seqlen, num_heads, head_dim = q.shape
        scale = 1.0 / math.sqrt(self.config.head_size)

        # print(f"sdpa {fa_is_available=}")
        if fa_is_available:
            # print(f"how??? {fa_meta_information=}")
            from .fa_padding import pad_input, upad_input

            if fa_meta_information is None:
                return flash_attn_func(
                    q=q,
                    k=k,
                    v=v,
                    # dropout_p=0.0, # FIXME for FA2
                    softmax_scale=scale,
                    causal=True,
                )
            else:
                cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k, indices = fa_meta_information

                # TODO: check if we qkv pack or index separately instead - my hunch says indexing is more expensive
                # GQA
                if q.size() != k.size():
                    k = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
                    v = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)

                # For reference: https://github.com/Dao-AILab/flash-attention/issues/432#issuecomment-1698610752
                # [bsz, seqlen, 3, num_heads, head_dim]
                qkv = torch.stack(tensors=(q, k, v), dim=2)

                # [bsz, seqlen, (3 * num_heads * head_dim)]
                qkv = rearrange(qkv, "b s three h d -> b s (three h d)")
                # [nnz, (3 * num_heads * head_dim)]
                qkv = upad_input(qkv, indices=indices, batch=batch, seqlen=seqlen)
                # [nnz, 3, num_heads, head_dim]
                qkv = rearrange(qkv, "nnz (three h d) -> nnz three h d", three=3, h=num_heads, d=head_dim)

                # 3 * [nnz, num_heads, head_dim]
                q, k, v = torch.unbind(qkv, dim=1)
                seqused_q = seqused_q.to(torch.int32)
                seqused_k = seqused_k.to(torch.int32)

                # [nnz, num_heads, head_dim]
                y = flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    seqused_q=seqused_q,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    # dropout_p=0.0, # FIXME for FA2
                    softmax_scale=scale,
                    causal=True,
                )

                # [bsz, seqlen, hidden_dim]
                y = pad_input(y, indices=indices, batch=batch, seqlen=seqlen)

                return y

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

        # torch has a bug for 2.1 ish on non-contiguous memory layouts
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # For compiling make it non-inline + add seq_len restriction (full attention on q)
        is_causal = True if mask is None and seqlen > 1 else False

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            # dropout_p=0.0, # FIXME for FA2
            scale=scale,
            is_causal=is_causal,
        )

        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.swiglu = SwiGLU(config.n_embd, config.intermediate_size, bias=False, _pack_weights=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_fc_1 = self.fc_1(x)
        # x_fc_2 = self.fc_2(x)
        # x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        # return self.proj(x)
        return self.swiglu(x)


class xformersfree_LLaMAMLP(nn.Module):
    # TODO use
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.w3 = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.w1(x)
        x_fc_2 = self.w2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.w3(x)


def build_rope_cache(
        seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """

    # RoPE in bfloat16 suffers from a significant performance downgrade --> we need to force float32 in any case
    # See https://github.com/huggingface/transformers/pull/29285 for reference
    device_type = device.type if isinstance(device.type, str) else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float32, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device) / condense_ratio

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta)

        cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2:]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
