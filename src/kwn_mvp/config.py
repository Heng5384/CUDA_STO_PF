"""Configuration loading and deterministic configuration provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


class ConfigurationError(ValueError):
    """Raised when a KWN configuration is malformed or unsupported."""


@dataclass(frozen=True)
class ConfigDocument:
    """Parsed JSON-subset YAML document with its canonical SHA-256 digest."""

    path: Path
    data: Dict[str, Any]
    sha256: str


def canonical_json(data: Dict[str, Any]) -> str:
    """Return the stable JSON representation used for config hashing."""

    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(data: Dict[str, Any]) -> str:
    """Calculate the SHA-256 hash of canonical configuration content."""

    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> ConfigDocument:
    """Load a JSON-subset YAML configuration without an undeclared YAML dependency.

    JSON is a valid YAML subset.  The repository deliberately stores MVP
    configuration files in that subset because the local Python runtime has no
    PyYAML installation.  This keeps parsing deterministic and dependency-free.
    """

    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"{config_path} must use the JSON subset of YAML; parsing failed at line "
            f"{exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Top-level configuration must be an object: {config_path}")
    return ConfigDocument(path=config_path, data=data, sha256=config_hash(data))


def require_mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a required mapping field or raise an actionable configuration error."""

    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Required mapping '{key}' is missing or not an object")
    return value


def require_number(data: Dict[str, Any], key: str, *, positive: bool = False) -> float:
    """Read a numeric field, optionally requiring strict positivity."""

    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise ConfigurationError(f"Required numeric field '{key}' is missing or invalid")
    numeric = float(value)
    if positive and numeric <= 0.0:
        raise ConfigurationError(f"Field '{key}' must be > 0; got {numeric!r}")
    return numeric
