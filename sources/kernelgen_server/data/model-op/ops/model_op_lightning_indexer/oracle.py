REFERENCE_DEVICE = 'target'

import math

import torch

QUERY_HEADS = 64
HEAD_DIM = 128
KEY_BLOCKS = 4422
BLOCK_SIZE = 128
BLOCK_TABLE_WIDTH = 260
TOP_K = 512

_INT64_MAX = 9223372036854775807


def run(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    return_value=False,
):
    """Lightning Indexer, FlagGems-vllm PR #799 semantics.

    Transcribed from the delivery's torch_reference_lightning_indexer. Per
    request, the paged key cache is gathered into a logical [key_length, D] view
    through block_table, and for each query token in that request:

        active_key_length = key_length - (query_end - token) + 1

    which is the causal cutoff. Non-positive lengths leave the row at its -1
    fill. When active_key_length <= sparse_count the row is exactly
    arange(active_key_length) -- the indexer degenerates to "take everything"
    and no scoring happens. Otherwise:

        qk     = query[token] @ logical_key[:active].T      # float32
        scores = (relu(qk) * weights[token][:, None]).sum(0)
        row    = topk(scores, sparse_count).indices

    Note the relu is applied to the raw QK before the per-head weighting, so
    negative weights still contribute negatively; that ordering matters.

    The second return value is an empty tensor of query's dtype -- the shipped
    reference returns `torch.empty((0,))` and the test asserts the pair's shape,
    dtype and device rather than any values.

    Index order within a row is not part of the contract: the source test
    compares `torch.sort(...).values` on both sides, so `valid` below compares
    per-row multisets. Ties in `scores` genuinely admit multiple valid answers,
    which is why this is not a plain elementwise comparison.
    """
    if layout_query != "TND" or layout_key != "PA_BSND" or sparse_mode != 3:
        raise NotImplementedError("only TND / PA_BSND / sparse_mode 3 are covered")
    if pre_tokens != _INT64_MAX or next_tokens != _INT64_MAX:
        raise NotImplementedError("only the default token windows are covered")

    output = torch.full(
        (query.shape[0], 1, sparse_count),
        -1,
        dtype=torch.int32,
        device=query.device,
    )
    query_ends = [int(v) for v in actual_seq_lengths_query.cpu().tolist()]
    key_lengths = [int(v) for v in actual_seq_lengths_key.cpu().tolist()]
    query_begin = 0
    for request, query_end in enumerate(query_ends):
        key_length = key_lengths[request]
        block_count = math.ceil(key_length / key.shape[1])
        if block_count:
            physical_blocks = block_table[request, :block_count].to(torch.long)
            logical_key = key.index_select(0, physical_blocks).reshape(
                -1, key.shape[-1]
            )
            logical_key = logical_key[:key_length].to(torch.float32)
        else:
            logical_key = key.new_empty((0, key.shape[-1]), dtype=torch.float32)

        for token in range(query_begin, query_end):
            active_key_length = key_length - (query_end - token) + 1
            if active_key_length <= 0:
                continue
            if active_key_length <= sparse_count:
                output[token, 0, :active_key_length] = torch.arange(
                    active_key_length, dtype=torch.int32, device=query.device
                )
                continue

            qk = torch.matmul(
                query[token].to(torch.float32),
                logical_key[:active_key_length].transpose(0, 1),
            )
            scores = (
                torch.relu(qk) * weights[token].to(torch.float32).unsqueeze(1)
            ).sum(dim=0)
            output[token, 0] = torch.topk(scores, sparse_count).indices.to(torch.int32)
        query_begin = query_end

    return output, torch.empty((0,), dtype=query.dtype, device=query.device)


def valid(ref_outputs, sol_outputs, inputs, ctx):
    """Per-row multiset comparison, mirroring assert_index_multisets_equal.

    The indexer's row order is unspecified (the kernel emits proposals in merge
    order, not score order), so the shipped test sorts both sides along the last
    dim and requires an exact match -- rtol=0, atol=0. Structure, dtype and
    shape are already checked by the harness before this runs; the empty
    `values` leaf carries no data to compare.
    """
    ref_indices, sol_indices = ref_outputs[0], sol_outputs[0]
    ref_sorted = torch.sort(ref_indices, dim=-1).values
    sol_sorted = torch.sort(sol_indices, dim=-1).values
    if not torch.equal(ref_sorted, sol_sorted):
        mismatched = (ref_sorted != sol_sorted).any(dim=-1).sum().item()
        total = ref_sorted.reshape(-1, ref_sorted.shape[-1]).shape[0]
        return {
            "passed": False,
            "message": (
                f"selected index multiset differs on {mismatched} of {total} rows"
            ),
            "metrics": {"mismatched_rows": float(mismatched)},
        }
    return {"passed": True, "message": "", "metrics": {"mismatched_rows": 0.0}}


# Case registry transcribed from tests/lightning_indexer_utils.py::CASES:
# case_id -> (query_t, local_q_lengths, key_lengths)
# `global_q_lengths` is registry bookkeeping that make_inputs never consumes.
_CASES = {
    "R1": (1, (1,), (8,)),
    "R2": (16, (16,), (128,)),
    "R3": (256, (256,), (2048,)),
    "R4": (1024, (1024,), (8192,)),
    "R5": (1, (1, 0, 0, 0), (130, 0, 0, 0)),
    "R6": (1, (1, 0, 0, 0), (2050, 0, 0, 0)),
    "R7": (1, (1, 0, 0, 0), (8194, 0, 0, 0)),
    "R8": (33, (1, 16, 16), (130, 128, 128)),
    "R9": (
        81,
        (1, 0, 0, 16, 16, 16, 16, 16),
        (132, 130, 130, 128, 128, 128, 128, 128),
    ),
    "R10": (
        2,
        (1, 1, 0, 0, 0, 0, 0, 0),
        (134, 132, 132, 130, 130, 130, 130, 130),
    ),
    "R11": (
        2,
        (1, 1, 0, 0, 0, 0, 0, 0),
        (136, 136, 136, 136, 136, 0, 0, 0),
    ),
    "R12": (1, (1, 0), (130, 2050)),
}


def _scalars(ctx):
    """Unwrap ctx["inputs"] recipe entries to plain values."""
    out = {}
    for name, raw in ctx["inputs"].items():
        if isinstance(raw, dict) and "value" in raw:
            out[name] = raw["value"]
        else:
            out[name] = raw
    return out


def gen_inputs(ctx, device):
    """Build one registered indexer case.

    Transcribed from tests/lightning_indexer_utils.py::make_inputs. The paged
    key cache is a fixed KEY_BLOCKS x BLOCK_SIZE arena and the block table hands
    out consecutive physical blocks per request, so requests do not alias; the
    query lengths are cumulative while the key lengths are per-request, which is
    the asymmetry the reference's `query_end` bookkeeping relies on.
    """
    spec = _scalars(ctx)
    query_t, local_q_lengths, key_lengths = _CASES[spec["case"]]
    sparse_count = int(spec.get("sparse_count", TOP_K))

    torch.manual_seed(int(ctx["seed"]))
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(int(ctx["seed"]))

    query = torch.empty(
        (query_t, QUERY_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-8, 8)
    key = torch.empty(
        (KEY_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-8, 8)
    weights = torch.empty(
        (query_t, QUERY_HEADS), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)

    cumulative, total = [], 0
    for length in local_q_lengths:
        total += int(length)
        cumulative.append(total)
    query_lengths = torch.tensor(cumulative, dtype=torch.int32, device=device)
    key_lengths_t = torch.tensor(key_lengths, dtype=torch.int32, device=device)

    block_table = torch.zeros(
        (len(key_lengths), BLOCK_TABLE_WIDTH), dtype=torch.int32, device=device
    )
    next_block = 0
    for request, key_length in enumerate(key_lengths):
        block_count = math.ceil(key_length / BLOCK_SIZE)
        if next_block + block_count > KEY_BLOCKS:
            raise ValueError("block table exceeds the allocated key cache")
        if block_count:
            block_table[request, :block_count] = torch.arange(
                next_block, next_block + block_count, dtype=torch.int32, device=device
            )
        next_block += block_count

    return {
        "query": query,
        "key": key,
        "weights": weights,
        "actual_seq_lengths_query": query_lengths,
        "actual_seq_lengths_key": key_lengths_t,
        "block_table": block_table,
        "layout_query": "TND",
        "layout_key": "PA_BSND",
        "sparse_count": sparse_count,
        "sparse_mode": 3,
    }
