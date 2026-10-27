"""Portable locations for external experiment assets and inference-only runs.

RECOVERY_CONFIG optionally names a JSON object. Relative configuration paths
are resolved against that file's directory. Environment overrides are
RECOVERY_DATA_ROOT and RECOVERY_RUN_DIR. No assets are downloaded here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_config_file = os.environ.get("RECOVERY_CONFIG")
_config_path = Path(_config_file).expanduser().resolve() if _config_file else None
_base = _config_path.parent if _config_path else PROJECT_ROOT
_config = json.loads(_config_path.read_text()) if _config_path else {}
if not isinstance(_config, dict):
    raise ValueError("RECOVERY_CONFIG must contain a JSON object")


def _path(value, base=_base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config():
    return dict(_config)


DATA_ROOT = _path(os.environ.get("RECOVERY_DATA_ROOT", _config.get("data_root", PROJECT_ROOT)))
RUN_DIR = _path(os.environ.get("RECOVERY_RUN_DIR", _config.get("run_dir", PROJECT_ROOT / "results/workshop_functional_recovery_v1")))
LOCAL_FILES_ONLY = _config.get("local_files_only", True)
if not isinstance(LOCAL_FILES_ONLY, bool):
    raise ValueError("local_files_only must be a JSON boolean")


def configure_hf_environment():
    if _config.get("hf_cache"):
        os.environ["HF_HUB_CACHE"] = str(_path(_config["hf_cache"]))
    if LOCAL_FILES_ONLY:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


configure_hf_environment()


def model_path(kind):
    """Return a configured local path or a cached Hugging Face identifier."""
    defaults = {"recovery": "Qwen/Qwen3.6-27B", "victim": "meta-llama/Meta-Llama-3.1-8B-Instruct"}
    value = str(_config.get(f"{kind}_model", defaults[kind]))
    if value.startswith(("/", "~", ".")) or (_base / value).exists():
        return str(_path(value))
    return value


def adapter_path(category):
    overrides = _config.get("adapters", {})
    if category in overrides:
        return _path(overrides[category])
    root = _path(_config["adapter_root"]) if "adapter_root" in _config else DATA_ROOT / "results"
    return root / f"sixcat200_{category}_structured_qwen36_27b_lora/adapter"


def cache_dir():
    value = _config.get("hf_cache") or os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
    return str(_path(value)) if value else None


def source_key(path):
    """Hash manifests use data-relative paths, or absolute external checkpoints."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(DATA_ROOT))
    except ValueError:
        return str(path)
