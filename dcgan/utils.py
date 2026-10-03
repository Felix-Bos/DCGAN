"""Utilitaires : configuration YAML, device, seed et checkpoints."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_override(cfg: dict, dotted_key: str, raw_value: str):
    """Applique une surcharge 'train.epochs=10' (la valeur est parsée en YAML)."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw_value)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_dirs(cfg: dict) -> dict:
    base = Path(cfg["output"]["dir"]) / cfg["experiment_name"]
    dirs = {
        "base": base,
        "tensorboard": base / "tensorboard",
        "checkpoints": base / "checkpoints",
        "samples": base / "samples",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_checkpoint(path, **state):
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def latest_checkpoint(ckpt_dir) -> Path | None:
    ckpts = sorted(Path(ckpt_dir).glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def prune_checkpoints(ckpt_dir, keep_last):
    if not keep_last:
        return
    ckpts = sorted(Path(ckpt_dir).glob("epoch_*.pt"))
    for old in ckpts[:-keep_last]:
        old.unlink()
