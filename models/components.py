import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias),
            # nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        og_shape = x.shape
        x = self.model(x.view(-1, og_shape[-1]))
        return x.view(*og_shape[:-1], -1)

