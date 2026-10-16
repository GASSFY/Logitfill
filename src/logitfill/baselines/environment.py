"""Versions recorded beside a baseline result."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch


def config_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def runtime_environment(attn_impl: str, dtype: str) -> dict:
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "python_prefix": sys.prefix,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "transformers": transformers.__version__,
        "attn_impl": attn_impl,
        "dtype": dtype,
    }
