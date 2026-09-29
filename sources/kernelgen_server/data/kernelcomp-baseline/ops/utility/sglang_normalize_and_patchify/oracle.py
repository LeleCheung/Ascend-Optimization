REFERENCE_DEVICE = 'target'

import torch
def run(image, image_scale, image_bias, patch_size, padded_height, padded_width):
    pad_h = padded_height - image.shape[-2]
    pad_w = padded_width - image.shape[-1]
    if pad_h > 0 or pad_w > 0:
        # Pad first: the padded pixels are normalized too, so they carry the
        # bias rather than zero.
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=0.0)

    normalized = torch.addcmul(image_bias, image, image_scale)

    b, c, h, w = normalized.shape
    gh, gw = h // patch_size, w // patch_size
    tiled = normalized.view(b, c, gh, patch_size, gw, patch_size)
    return tiled.permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c, patch_size, patch_size)
