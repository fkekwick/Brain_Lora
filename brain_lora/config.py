"""Small, strict YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a YAML mapping.")
    return config


def get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def require(config: dict[str, Any], *keys: str) -> list[Any]:
    missing = [key for key in keys if get(config, key) in (None, "")]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required configuration value(s): {joined}")
    return [get(config, key) for key in keys]


def require_file(config: dict[str, Any], key: str) -> Path:
    (raw,) = require(config, key)
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ConfigError(f"Configuration value '{key}' is not a file: {path}")
    return path


def require_dir(config: dict[str, Any], key: str) -> Path:
    (raw,) = require(config, key)
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise ConfigError(f"Configuration value '{key}' is not a directory: {path}")
    return path


def output_dir(config: dict[str, Any], key: str = "output.dir") -> Path:
    (raw,) = require(config, key)
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
