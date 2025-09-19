# ortho/ae_train.py
import os, time, json, copy, argparse
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .data_io import load_species_matrix, fit_scaler_train_only, save_scaler, make_train_val_indices, build_dataloaders, save_json
from .ae_models import AE, cosine_recon_loss
from .dec_losses import student_t_assign, target_distribution, kl_p_q, init_dec_centers

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def build_exp_dir(cfg: Dict[str, Any]) -> str:
    ts = _timestamp()
    data = cfg["data"]; ae = cfg["ae"]; dec = cfg.get("dec", {})
    model_name = data["model_name"]; L = ae["latent_dim"]
    loss_tag  = ae["loss_type"]
    lam_tag   = f"L1-{ae['lambda_l1']}"
    dec_tag   = "DEC-on" if dec.get("use", False) else "DEC-off"
    name = (
        f"TRAIN[{'+'.join(data['train_spec'])}]_TEST[{data['test_spec']}]_"
        f"{model_name}_AE{L}_{loss_tag}_{lam_tag}_{dec_tag}_{ts}"
    )
    root = cfg["run"]["results_root"]
    exp_dir = os.path.join(root, name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def train(cfg: Dict[str, Any]) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load pooled train ----------
    data_root = cfg["data"]["root"]
    train_spec = cfg["data"]["train_spec"]
    model_name = cfg["data"]["model_name"]
    emb_layer  = cfg["data"]["emb_layer"]

    Xs_list, meta_list = [], []
    for sp in train_spec:
        Xs_sp, ids_sp, meta_sp = load_species_matrix(sp, model_name, emb_layer, data_root)
        Xs_list.append(Xs_sp); meta_list += meta_sp
    X_pool = np.concatenate(Xs_list, axis=0).astype(np.float32)
    N_all, D = X_pool.shape

    # ---------- Scaler fit on train only ----------
    sc = fit_scaler_train_only(X_pool)
    exp_dir = build_exp_dir(cfg)
    save_scaler(sc, os.path.join(exp_dir, "scaler.joblib"))
    X_scaled = sc.transform(X_pool).astype(np.float32)

    # ---------- Splits ----------
    seed   = cfg["run"].get("rng_seed", 0)
    tr_idx, va_idx = make_train_val_indices(N_all, val_frac=cfg["ae"].get("val_frac", 0.10), seed=seed)
    tr_dl, va_dl = build_dataloaders(X_scaled, tr_idx, va_idx, cfg["ae"]["batch_size"])

    # ---------- Model / Optim ----------
    ae_cfg = cfg["ae"]
    ae = AE(D, ae_cfg["latent_dim"], dropout=ae_cfg.get("dropout", 0.1)).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=ae_cfg["lr"], weight_decay=ae_cfg["wd"])
    mse = nn.MSELoss()
    loss_type = ae_cfg["loss_type"].lower()
    lambda_l1 = float(ae_cfg["lambda_l1"])
    noise_sigma = float(ae_cfg.get("noise_sigma", 0.0))

    # ---------- DEC knobs ----------
    dec_cfg = cfg.get("dec", {})
    USE_DEC = bool(dec_cfg.get("use", False))
    DEC_ALPHA = float(dec_cfg.get("alpha", 1.0))
    LAMBDA_DEC = float(dec_cfg.get("lambda_dec", 0.5))
    DEC_WARMUP_EPOCHS = int(dec_cfg.get("warmup_epochs", 10))
    DEC_UPDATE_EVERY  = int(dec_cfg.get("update_every", 5))
    if USE_DEC:
        # choose a training K (independent of eval K)
        mode = dec_cfg.get("k_mode", "n2").lower()
        if mode == "ogs":
            # training OG count over pooled meta
            from .data_io import count_OGs
            DEC_K = max(2, count_OGs(meta_list))
        else:
            DEC_K = max(2, len(tr_idx)//2)
    else:
        DEC_K = 0

    # ---------- Training loop ----------
    best_val = float("inf"); best_state = None; no_improve = 0
    mu = None              # DEC centers (nn.Parameter when created)
    P_full = None          # [N_all, DEC_K]

    EPOCHS  = int(ae_cfg["epochs"])
    PATIENCE= int(ae_cfg["patience"])

    print(f"[AE] start | device={device} | D={D}→L={ae_cfg['latent_dim']} | loss={loss_type} | L1={lambda_l1}")
    if USE_DEC:
        print(f"[DEC] on | K={DEC_K} | warmup={DEC_WARMUP_EPOCHS} | update_every={DEC_UPDATE_EVERY} | λ_dec={LAMBDA_DEC}")

    for ep in range(1, EPOCHS+1):
        ae.train(); tr_loss = tr_recon = 0.0

        # Init DEC centers after warmup
        if USE_DEC and (ep == DEC_WARMUP_EPOCHS + 1):
            mu = torch.nn.Parameter(init_dec_centers(ae, X_scaled, DEC_K, device=str(device), seed=seed))
            opt = torch.optim.AdamW(list(ae.parameters()) + [mu], lr=ae_cfg["lr"], weight_decay=ae_cfg["wd"])

        # Refresh P_full every DEC_UPDATE_EVERY epochs
        if USE_DEC and (mu is not None) and ((ep - (DEC_WARMUP_EPOCHS + 1)) % DEC_UPDATE_EVERY == 0):
            ae.eval()
            P_full = torch.zeros((N_all, DEC_K), dtype=torch.float32, device=device)
            with torch.no_grad():
                dl_all = DataLoader(NPDataset=X_scaled, batch_size=1024, shuffle=False)  # placeholder typing comment
            # build a small loader on-the-fly
            from torch.utils.data import DataLoader as _DL
            from .data_io import NPDataset as _DS
            dl_all = _DL(_DS(X_scaled), batch_size=1024, shuffle=False, drop_last=False)
            with torch.no_grad():
                for idx, xb in dl_all:
                    xb = torch.from_numpy(xb).to(device)
                    _, z = ae(xb)
                    q = student_t_assign(z, mu, alpha=DEC_ALPHA)
                    P_full[idx.to(device)] = target_distribution(q)
            ae.train()

        # --------- train over batches ---------
        for idx, xb in tr_dl:
            xb = torch.from_numpy(xb).to(device) if isinstance(xb, np.ndarray) else xb.to(device)
            noisy = xb + noise_sigma * torch.randn_like(xb) if noise_sigma > 0 else xb

            x_hat, z = ae(noisy)
            if loss_type == "mse":
                recon = mse(x_hat, xb)
            elif loss_type == "cosine":
                recon = cosine_recon_loss(x_hat, xb)
            else:
                raise ValueError("loss_type must be 'mse' or 'cosine'")

            spars = z.abs().mean()
            loss  = recon + lambda_l1 * spars

            if USE_DEC and (mu is not None) and (P_full is not None):
                q = student_t_assign(z, mu, alpha=DEC_ALPHA)
                p = P_full[idx.to(P_full.device)]
                L_dec = kl_p_q(p, q)
                loss  = loss + LAMBDA_DEC * L_dec

            opt.zero_grad(); loss.backward(); opt.step()

            tr_loss  += loss.item()  * xb.size(0)
            tr_recon += recon.item() * xb.size(0)

        tr_loss  /= len(tr_dl.dataset)
        tr_recon /= len(tr_dl.dataset)

        # --------- validation (no DEC term) ---------
        ae.eval(); va_loss = va_recon = 0.0
        with torch.no_grad():
            for idx, xb in va_dl:
                xb = torch.from_numpy(xb).to(device) if isinstance(xb, np.ndarray) else xb.to(device)
                x_hat, z = ae(xb)
                recon = mse(x_hat, xb) if loss_type == "mse" else cosine_recon_loss(x_hat, xb)
                spars = z.abs().mean()
                loss  = recon + lambda_l1 * spars
                va_loss  += loss.item()  * xb.size(0)
                va_recon += recon.item() * xb.size(0)
        va_loss  /= len(va_dl.dataset)
        va_recon /= len(va_dl.dataset)

        tag = ""
        # you prefer best by val-total
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {"ae": copy.deepcopy(ae.state_dict())}
            if USE_DEC and (mu is not None):
                best_state["mu"] = mu.detach().cpu()
            tag = "**best**"; no_improve = 0
        else:
            no_improve += 1

        val_l1 = max(0.0, va_loss - va_recon)
        print(f"[AE] ep{ep:03d} train: total={tr_loss:.5f} ({loss_type}={tr_recon:.5f}) | "
              f"val: total={va_loss:.5f} ({loss_type}={va_recon:.5f}, l1={val_l1:.5f}) | {tag}")

        if no_improve >= PATIENCE:
            break

    # ---------- save checkpoint ----------
    ck = {"ae": best_state["ae"]}
    if USE_DEC and ("mu" in best_state): ck["mu"] = best_state["mu"]
    loss_tag = ae_cfg["loss_type"].lower()
    lam_tag  = f"lam{ae_cfg['lambda_l1']}"
    torch.save(ck, os.path.join(exp_dir, f"ae_checkpoint_{loss_tag}_{lam_tag}.pt"))

    # meta
    save_json({
        "cfg": cfg, "exp_dir": exp_dir, "device": str(device),
        "dims": {"D": int(D), "L": int(ae_cfg["latent_dim"])}
    }, os.path.join(exp_dir, "run_meta.json"))

    print(f"[saved] → {exp_dir}")
    return exp_dir


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args, overrides = ap.parse_known_args()

    cfg = load_yaml(args.config)

    # simple CLI override: key=value (e.g., ae.lambda_l1=7.5e-4)
    for kv in overrides:
        if "=" in kv:
            key, val = kv.split("=", 1)
            # walk & set
            d = cfg
            ks = key.split(".")
            for k in ks[:-1]:
                d = d.setdefault(k, {})
            # cast numbers if possible
            try:
                v = json.loads(val)
            except Exception:
                v = val
            d[ks[-1]] = v

    # defaults for roots
    cfg.setdefault("data", {}).setdefault("root", os.environ.get("ORTHO_DATA_ROOT", f"/scratch/{os.environ.get('USER','user')}/datasets/OrthoLM"))
    cfg.setdefault("run", {}).setdefault("results_root", f"/scratch/{os.environ.get('USER','user')}/ortho_runs/results")

    train(cfg)

if __name__ == "__main__":
    main()
