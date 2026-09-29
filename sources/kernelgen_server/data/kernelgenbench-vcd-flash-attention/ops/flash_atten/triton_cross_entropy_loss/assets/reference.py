# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

try:
    from flash_attn.ops.triton import cross_entropy as _m
except (ModuleNotFoundError, ImportError):
    _m = None


def triton_cross_entropy_loss(
    logits,
    labels,
    precomputed_lse=None,
    label_smoothing=0.0,
    logit_scale=1.0,
    lse_square_scale=0.0,
    ignore_index=-100,
    inplace_backward=False,
    process_group=None,
):
    """FlashAttention Triton baseline for cross_entropy_loss.

    Fused softmax cross-entropy with optional label smoothing and z-loss:
        loss = NLL(softmax(logit_scale * logits), labels)
               + lse_square_scale * logsumexp(logit_scale * logits)^2

    Args:
        logits (Tensor): unnormalized scores, shape (batch, vocab_size).
        labels (Tensor): int64 target indices, shape (batch,).
        precomputed_lse (Tensor|None): optional precomputed log-sum-exp.
        label_smoothing (float): label smoothing factor.
        logit_scale (float): scale applied to logits before the loss.
        lse_square_scale (float): z-loss coefficient (0.0 disables).
        ignore_index (int): label value whose loss is forced to 0.
        inplace_backward (bool): compute backward in-place on logits.
        process_group (ProcessGroup|None): tensor-parallel vocab group.

    Returns:
        (losses, z_losses): both shape (batch,), float.
    """
    return _m.cross_entropy_loss(
        logits, labels, precomputed_lse=precomputed_lse,
        label_smoothing=label_smoothing, logit_scale=logit_scale,
        lse_square_scale=lse_square_scale, ignore_index=ignore_index,
        inplace_backward=inplace_backward, process_group=process_group,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: triton_cross_entropy_loss requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: triton_cross_entropy_loss requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        B, V = 128, 4096
        logits = torch.randn(B, V, device=device, dtype=dtype)
        labels = torch.randint(0, V, (B,), device=device, dtype=torch.int64)

        # ---- plain cross entropy (no z-loss, no smoothing) ----
        losses, z_losses = triton_cross_entropy_loss(logits, labels)
        ref = torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="none")
        torch.testing.assert_close(losses.float(), ref, rtol=2e-2, atol=2e-2)
        print(f"pass: triton_cross_entropy_loss dtype={dtype}")

        # ---- with z-loss ----
        zl = 1e-4
        losses2, z2 = triton_cross_entropy_loss(logits, labels, lse_square_scale=zl)
        lse = torch.logsumexp(logits.float(), dim=-1)
        ref2 = ref + zl * lse ** 2
        torch.testing.assert_close(losses2.float(), ref2, rtol=2e-2, atol=2e-2)
        print(f"pass: triton_cross_entropy_loss z-loss dtype={dtype}")

    print("\ntriton_cross_entropy_loss all tests passed")
