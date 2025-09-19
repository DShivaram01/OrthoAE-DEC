# ortho/ae_models.py
import torch
import torch.nn as nn

class AE(nn.Module):
    """
    Simple MLP AE: D -> 512 -> 128 -> L  ||  L -> 128 -> 512 -> D
    """
    def __init__(self, in_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128),   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512),        nn.ReLU(),
            nn.Linear(512, in_dim)
        )

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z


def cosine_recon_loss(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    xh = x_hat / (x_hat.norm(dim=1, keepdim=True) + eps)
    xx = x     / (x.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (xh * xx).sum(dim=1).mean()
