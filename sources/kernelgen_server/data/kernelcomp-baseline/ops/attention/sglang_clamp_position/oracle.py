REFERENCE_DEVICE = 'target'

import torch
def run(seq_lens):
    return (seq_lens - 1).clamp(min=0)
