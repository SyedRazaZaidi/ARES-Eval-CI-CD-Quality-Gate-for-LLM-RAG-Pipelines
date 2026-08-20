"""Repository path helpers that work from source checkout or editable install."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "data").exists():
            return parent
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    return here.parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def config_dir() -> Path:
    return repo_root() / "config"


def artifacts_dir() -> Path:
    path = repo_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ares_home() -> Path:
    path = repo_root() / ".ares"
    path.mkdir(parents=True, exist_ok=True)
    return path
