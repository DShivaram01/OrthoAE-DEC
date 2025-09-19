# ortho/utils_fs.py
import os, time, json
from typing import Dict, Any, List

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def build_exp_name(cfg: Dict[str, Any]) -> str:
    data = cfg["data"]; ae = cfg["ae"]; dec = cfg.get("dec", {})
    model_name = data["model_name"]; L = ae["latent_dim"]
    loss_tag  = ae["loss_type"].lower()
    lam_tag   = f"L1-{ae['lambda_l1']}"
    dec_tag   = "DEC-on" if dec.get("use", False) else "DEC-off"
    return (
        f"TRAIN[{'+'.join(data['train_spec'])}]_TEST[{data['test_spec']}]_"
        f"{model_name}_AE{L}_{loss_tag}_{lam_tag}_{dec_tag}_{timestamp()}"
    )

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True); return path

def save_json(obj, path: str) -> None:
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def set_thread_env(n_threads: int):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
