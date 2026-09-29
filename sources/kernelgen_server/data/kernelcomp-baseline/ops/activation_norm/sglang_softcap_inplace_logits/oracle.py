REFERENCE_DEVICE = 'target'

import torch
def run(full_logits, final_logit_softcapping):
    return (full_logits / final_logit_softcapping).tanh() * final_logit_softcapping
