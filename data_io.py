# ortho/data_io.py
import os, json
from typing import List, Tuple, Dict, Any, Iterable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import esm
except Exception as e:
    esm = None

# -------------------------- FASTA + embedding IO --------------------------

def load_species_matrix(
    species: str,
    model_name: str,
    emb_layer: int,
    data_root: str,
) -> Tuple[np.ndarray, List[str], List[List[str]]]:
    """
    Returns:
        Xs   : (N, D) float32 mean embeddings
        ids  : list[str] fasta headers (order matches rows)
        meta : list[list[str]] header split by '|'
               Keep exact parsing: [species, protein_id, organism, desc, OG_ID]
    Conventions:
      FASTA:        {data_root}/{species}.fasta
      Embeddings:   {data_root}/{species}_emb_{model_name}/{header}.pt
      Tensor key:   embs['mean_representations'][emb_layer]
    """
    fasta_path = os.path.join(data_root, f"{species}.fasta")
    emb_dir    = os.path.join(data_root, f"{species}_emb_{model_name}")
    if esm is None:
        raise RuntimeError("esm not available. Please `pip install fair-esm`.")

    Xs, ids, meta = [], [], []
    for header, _seq in esm.data.read_fasta(fasta_path):
        pt = os.path.join(emb_dir, f"{header}.pt")
        if not os.path.isfile(pt):
            continue
        embs = torch.load(pt, map_location="cpu")
        Xs.append(embs["mean_representations"][emb_layer])
        ids.append(header)
        meta.append(header.split("|"))
    if not Xs:
        raise RuntimeError(f"No embeddings found under {emb_dir}")
    Xs = torch.stack(Xs, dim=0).numpy().astype(np.float32)
    return Xs, ids, meta


def count_OGs(prot_meta: List[List[str]]) -> int:
    return len(set([p[4].strip() for p in prot_meta]))


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -------------------------- Scaler & splits --------------------------------

def fit_scaler_train_only(X_pool: np.ndarray) -> StandardScaler:
    sc = StandardScaler().fit(X_pool)
    return sc

def save_scaler(sc: StandardScaler, out_path: str) -> None:
    joblib.dump(sc, out_path)

def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)

def make_train_val_indices(N: int, val_frac: float = 0.10, seed: int = 0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(val_frac * N))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return train_idx, val_idx


# -------------------------- Datasets & loaders -----------------------------

class NPDataset(Dataset):
    """
    Index-aware dataset: returns (global_index, vector) so DEC can align targets.
    """
    def __init__(self, X: np.ndarray, index: np.ndarray | None = None):
        self.X = X.astype(np.float32, copy=False)
        if index is None:
            self.index = np.arange(self.X.shape[0], dtype=np.int64)
        else:
            self.index = np.asarray(index, dtype=np.int64)

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        return j, self.X[j]


def build_dataloaders(
    X_scaled: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    tr = DataLoader(
        NPDataset(X_scaled, train_idx),
        batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    va = DataLoader(
        NPDataset(X_scaled, val_idx),
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return tr, va


# ------------------------- Geometry helpers --------------------------------

def l2_normalize_rows(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

@torch.no_grad()
def torch_l2_normalize_rows(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    n = x.norm(dim=1, keepdim=True)
    return x / (n + eps)


# ------------------------- k-list helpers -----------------------------------

def compute_k_list_from_tokens(meta: List[List[str]], N: int, tokens: Iterable) -> Tuple[List[int], int]:
    ogs = count_OGs(meta)
    out: List[int] = []
    seen = set()
    for t in tokens:
        if isinstance(t, int):
            k = int(t)
        elif isinstance(t, str):
            t_up = t.upper()
            if t_up in ("N2", "N//2", "N_HALF"):
                k = max(2, N // 2)
            elif t_up in ("OGS", "#OG", "#OGS"):
                k = max(2, ogs)
            else:
                raise ValueError(f"Unknown k token: {t}")
        else:
            raise ValueError(f"Unsupported k token type: {type(t)}")

        if 1 < k <= N and k not in seen:
            seen.add(k)
            out.append(k)
    return out, ogs
