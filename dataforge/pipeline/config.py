from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name": "default",
    "inputs": ["data/samples/corpus.jsonl"],
    "output_dir": "outputs/default",
    "seed": 42,
    "extract": {"backend": "builtin", "min_chars": 50},
    "lang": {"backend": "heuristic", "fasttext_model": None, "keep": ["en", "ur"], "min_score": 0.5},
    # Gopher's 0.8 alphabetic-word floor removes math-heavy educational text (see configs/default.yaml).
    "heuristics": {"enabled": True, "min_alpha_word_frac": 0.6},
    "dedup": {
        "exact": True,
        "near": True,
        "mode": "streaming",
        "threshold": 0.7,
        "num_perm": 128,
        "ngram": 5,
        "char_level": False,
        "verify": "minhash",
        "workers": 1,
        "parallel_backend": "multiprocessing",
        "backend": "python",
    },
    "quality": {
        "enabled": True,
        "model_path": "models/edu_quality.pkl",
        "train_data": "data/samples/quality_train.jsonl",
        "thresholds": {"default": 2.5},
    },
    "safety": {
        "pii": {"enabled": True, "actions": {}, "max_pii_per_100_words": 5.0, "enable_names": True},
        "toxicity": {"enabled": True, "action": "drop", "threshold": 0.5, "classifier_path": None},
    },
    "export": {"formats": ["jsonl", "parquet"], "write_rejected": True, "by_language": True, "drop_gold": True},
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars/lists."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    return apply_overrides(cfg, overrides or [])


def resolve(path: str, base_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base_dir / p)
