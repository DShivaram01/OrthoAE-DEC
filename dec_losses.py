# ortho/dec_losses.py
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from .data_io import NPDataset

def student_t_assign(z: torch.Tensor, mu: torch.Tensor, alpha: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """
    z: [B,L], mu: [K,L]
    q_ij ∝ (1 + ||z_i - μ_j||^2 / α)^(-(α+1)/2)
    """
    diff  = z[:, None, :] - mu[None, :, :]
    dist2 = (diff * diff).sum(dim=-1)
    num   = (1.0 + dist2 / alpha).pow(-(alpha + 1.0) / 2.0)
    q     = num / (num.sum(dim=1, keepdim=True) + eps)
    return q

def target_distribution(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    p_ij = (q_ij^2 / f_j) / sum_l(q_il^2 / f_l), f_j = sum_i q_ij
    """
    f = q.sum(dim=0, keepdim=True) + eps
    p = (q * q) / f
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    return p

def kl_p_q(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=1).mean()

@torch.no_grad()
def init_dec_centers(
    ae,
    X_scaled: np.ndarray,
    k: int,
    device: str = "cuda",
    seed: int = 0,
    bs: int = 1024
) -> torch.Tensor:
    """
    Run a one-off KMeans on current latents to initialize DEC centers.
    Returns centers as a torch.FloatTensor [K, L] on device.
    """
    ds = NPDataset(X_scaled)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False)
    zs = []
    ae.eval()
    for idx, xb in dl:
        xb = torch.from_numpy(xb).to(device)
        _, z = ae(xb)
        zs.append(z.detach().cpu())
    Z = torch.cat(zs, dim=0).numpy().astype("float32")
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    return torch.from_numpy(km.cluster_centers_.astype("float32")).to(device)
