#!/usr/bin/env python
# scripts/run_train_ae.py
import sys, argparse
from ortho.config import load_yaml, apply_overrides, fill_defaults
from ortho.ae_train import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, overrides = ap.parse_known_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, overrides)
    cfg = fill_defaults(cfg)
    train(cfg)

if __name__ == "__main__":
    main()
