REFERENCE_DEVICE = 'target'

import torch
def run(residual, update, gate):
    product = (update.float() * gate.float()).to(residual.dtype)
    return (residual.float() + product.float()).to(residual.dtype)
