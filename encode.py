# ortho/encode.py
import os, argparse, json, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_io import load_species_matrix, load_scaler, l2_normalize_rows
from .ae_models import AE

try:
    import yaml
except Exception:
    yaml = None

def load_yaml(path): 
    if yaml is None: raise RuntimeError("PyYAML not installed.")
    with open(path, "r") as f: return yaml.safe_load(f)

@torch.no_grad()
def encode_latents(ae, scaler, X, device="cuda", batch_size=1024):
    Xs = scaler.transform(X).astype(np.float32)
    Zs = []
    ae.eval()
    for i in range(0, Xs.shape[0], batch_size):
        xb = torch.from_numpy(Xs[i:i+batch_size]).to(device)
        _, z = ae(xb)
        Zs.append(z.detach().cpu().numpy())
    Z = np.vstack(Zs).astype(np.float32)
    return Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_dir", required=True, help="directory with ae_checkpoint_... and scaler.joblib")
    ap.add_argument("--split", default="test", choices=["test"], help="currently encode test set")
    ap.add_argument("--row_l2", action="store_true", help="L2-normalize rows before saving")
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = cfg["data"]; ae_cfg = cfg["ae"]
    X, ids, meta = load_species_matrix(data["test_spec"], data["model_name"], data["emb_layer"], cfg["data"]["root"])
    D = X.shape[1]

    # find checkpoint & scaler
    scaler = load_scaler(os.path.join(args.exp_dir, "scaler.joblib"))
    ckpt_paths = sorted(glob.glob(os.path.join(args.exp_dir, "ae_checkpoint_*_lam*.pt")))
    if not ckpt_paths: raise RuntimeError("No ae_checkpoint_* found.")
    ckpt_path = ckpt_paths[0]  # or choose explicitly
    state = torch.load(ckpt_path, map_location=device)

    ae = AE(D, ae_cfg["latent_dim"], dropout=ae_cfg.get("dropout", 0.1)).to(device)
    ae.load_state_dict(state["ae"])

    Z = encode_latents(ae, scaler, X, device=device, batch_size=args.batch_size)
    if args.row_l2:
        Z = l2_normalize_rows(Z)

    out = os.path.join(args.exp_dir, f"{data['test_spec']}_latents_plain_(N{Z.shape[0]},L{Z.shape[1]}).npy")
    np.save(out, Z)
    # save ids/meta too
    np.save(os.path.join(args.exp_dir, f"{data['test_spec']}_ids.npy"), np.array(ids, dtype=object))
    with open(os.path.join(args.exp_dir, f"{data['test_spec']}_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
