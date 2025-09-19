# ortho/config.py
import os, json
from typing import Any, Dict, List

try:
    import yaml
except Exception:
    yaml = None

def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Overrides like: ["ae.lambda_l1=7.5e-4", "dec.use=false"]
    JSON-casts values when possible.
    """
    out = dict(cfg)
    for kv in overrides:
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        d = out
        keys = key.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            v = json.loads(val)
        except Exception:
            v = val
        d[keys[-1]] = v
    return out

def fill_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    user = os.environ.get("USER", "user")
    cfg.setdefault("data", {})
    cfg.setdefault("run", {})
    cfg["data"].setdefault("root", os.environ.get("ORTHO_DATA_ROOT", f"/scratch/{user}/datasets/OrthoLM"))
    cfg["run"].setdefault("results_root", f"/scratch/{user}/ortho_runs/results")
    cfg["run"].setdefault("rng_seed", 0)
    cfg.setdefault("cluster", {"n_init": 50, "max_iter": 500, "algo": "elkan", "seeds": [0,1,2], "ks": ["N2","OGs"]})
    return cfg
