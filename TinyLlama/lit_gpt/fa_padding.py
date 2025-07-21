# Adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py
# and from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)).reshape(
            -1, *other_shape
        )

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def upad_meta_information(input_ids, pad_id):
    """
    Arguments:
        input_ids: (batch, seqlen)
        pad_id: int
    Return:
        fa_varlen_meta_info: cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, indices
    """
    seqlen = input_ids.shape[-1]

    # get cu seq lens of each sample and max seq len
    seq_lens = torch.tensor(seqlen, dtype=torch.int32, device=input_ids.device).unsqueeze(0) - (
        input_ids == pad_id
    ).long().sum(dim=-1)
    cu_seqs = F.pad(seq_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
    max_seq_len = seq_lens.max().item()

    # indices of tokens we attend to
    indices = (input_ids.flatten() != pad_id).nonzero().long().squeeze(-1)

    return cu_seqs, cu_seqs, seq_lens, seq_lens, max_seq_len, max_seq_len, indices

def upad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        indices: (total_nnz,), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected (i.e. != pad token).
    """
    return index_first_axis(hidden_states.reshape(batch * seqlen, -1), indices)


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz,), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return output.reshape(batch, seqlen, -1)
