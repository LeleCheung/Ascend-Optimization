REFERENCE_DEVICE = 'target'

import torch
def run(g, chunk_size, reverse=False, scale=None):
    B, T, H = g.shape
    BT = chunk_size
    NT = T // BT

    g_c = g.float().view(B, NT, BT, H)
    if reverse:
        g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None:
        out = out * scale
    if reverse:
        out = out.flip(2)

    return out.reshape(B, T, H)
