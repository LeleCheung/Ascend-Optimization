REFERENCE_DEVICE = 'target'
import torch.nn.functional as F

import torch
def run(x, cache_x, padding):
    w_l, w_r, h_t, h_b, d_l, d_r = padding
    # The cache frames consume part of the temporal front padding.
    d_l = d_l - cache_x.shape[2]
    cat = torch.cat([cache_x, x], dim=2)
    # F.pad's order is (w_left, w_right, h_top, h_bottom, t_front, t_back).
    return torch.nn.functional.pad(cat, (w_l, w_r, h_t, h_b, d_l, d_r))
