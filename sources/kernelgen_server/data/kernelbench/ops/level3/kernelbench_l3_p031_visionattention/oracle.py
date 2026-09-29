REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)  # (seq_len, batch_size, embed_dim)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([embed_dim, num_heads])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
