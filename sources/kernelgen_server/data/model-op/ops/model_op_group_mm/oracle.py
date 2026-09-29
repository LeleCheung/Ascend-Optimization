REFERENCE_DEVICE = 'target'

import torch


def run(A, B, offs):
    """Grouped GEMM over a row-partitioned A, FlagGems group_mm semantics.

    Derived from the two kernels in src/flag_gems/ops/group_gemm.py
    (grouped_mm_kernel and grouped_mm_tma_kernel), which agree on the mapping:

        group_start = 0
        for group_idx in range(num_groups):
            group_end = offs[group_idx]
            # rows [group_start, group_end) of A times B[group_idx]
            offs_bk = group_idx * K          # B viewed as (num_groups * K, N)
            group_start = group_end

    so `offs` is an exclusive cumulative row boundary, not a per-group length.

    The accumulator is float32 with `allow_tf32=False` and the result is cast to
    C.dtype once at the end (`accumulator.to(C.dtype.element_ty)`), so the
    reference matmuls in float32 and casts once.

    C is allocated with `A.new_empty(M, N)` and the kernel only writes rows below
    offs[-1], so rows past the last boundary would be uninitialized garbage. The
    workloads therefore always set offs[-1] == M; this reference zero-fills any
    trailing rows so the result stays deterministic if that ever does not hold.
    """
    if A.dim() != 2:
        raise ValueError(f"A must be 2-D, got {A.dim()}-D")
    if B.dim() != 3:
        raise ValueError(f"B must be 3-D, got {B.dim()}-D")
    num_groups = int(offs.numel())
    if num_groups != B.shape[0]:
        raise ValueError(
            f"offs has {num_groups} entries but B has {B.shape[0]} groups"
        )
    m_rows, k_dim = A.shape
    n_cols = B.shape[2]
    out = A.new_zeros(m_rows, n_cols)
    bounds = [int(v) for v in offs.detach().cpu().tolist()]
    group_start = 0
    for group_idx, group_end in enumerate(bounds):
        if group_end > group_start:
            rows = A[group_start:group_end].to(torch.float32)
            mat = B[group_idx].to(torch.float32)
            out[group_start:group_end] = (rows @ mat).to(A.dtype)
        group_start = max(group_start, group_end)
    return out


def gen_inputs(ctx, device):
    """A, B and offs are interdependent, so the recipe cannot express them.

    `offs` must be strictly non-decreasing with offs[-1] == M, and B's group
    count must match offs.numel(), so all three are built together from the
    workload's group_sizes context.
    """
    # ctx["inputs"] holds raw recipe mappings, so unwrap each entry's "value"
    # before use; iterating the mapping itself would walk its keys instead.
    spec = {
        name: (raw["value"] if isinstance(raw, dict) and "value" in raw else raw)
        for name, raw in ctx["inputs"].items()
    }
    group_sizes = [int(v) for v in spec["group_sizes"]]
    k_dim = int(spec["K"])
    n_cols = int(spec["N"])
    dtype = getattr(torch, spec["dtype"])
    m_rows = sum(group_sizes)

    generator = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]))
    a = torch.randn(m_rows, k_dim, generator=generator, dtype=torch.float32)
    b = torch.randn(
        len(group_sizes), k_dim, n_cols, generator=generator, dtype=torch.float32
    )

    bounds = []
    running = 0
    for size in group_sizes:
        running += size
        bounds.append(running)

    return {
        "A": a.to(dtype).to(device),
        "B": b.to(dtype).to(device),
        "offs": torch.tensor(bounds, dtype=torch.int32, device=device),
    }
