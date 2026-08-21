"""Configuration loading for the IntelliOps platform.

Single source of truth: ``configs/config.yaml``. Environment variables win over
file values for anything deployment-specific (database URL, API keys, ports) so
the same image runs locally, in CI, and in the cloud without code changes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class Config:
    """Dict-backed config with dotted-path access and env-var overrides."""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT) -> None:
        self._data = data
        self.root = root

    # ------------------------------------------------------------------ access
    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted_key: str) -> Any:
        value = self.get(dotted_key, _MISSING)
        if value is _MISSING:
            raise KeyError(f"Missing config key: {dotted_key}")
        return value

    def path(self, dotted_key: str) -> Path:
        """Resolve a config value into an absolute path under the project root."""
        raw = self[dotted_key]
        p = Path(raw)
        return p if p.is_absolute() else (self.root / p)

    def ensure_dirs(self) -> None:
        for key in ("data_raw", "data_interim", "data_processed", "models", "reports", "figures"):
            self.path(f"paths.{key}").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- deployment
    @property
    def db_url(self) -> str:
        """Warehouse URL. ``INTELLIOPS_DB_URL`` overrides the config file."""
        env = os.getenv("INTELLIOPS_DB_URL")
        if env:
            return env
        url = self["warehouse.url"]
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
            # make the relative sqlite path absolute so cwd never matters
            rel = url.replace("sqlite:///", "", 1)
            abs_path = (self.root / rel).resolve()
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{abs_path}"
        return url

    @property
    def seed(self) -> int:
        return int(self.get("project.random_seed", 42))


_MISSING = object()


@lru_cache(maxsize=4)
def load_config(config_path: str | os.PathLike[str] | None = None) -> Config:
    """Load and cache the platform config."""
    path = Path(config_path or os.getenv("INTELLIOPS_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data, root=PROJECT_ROOT)
