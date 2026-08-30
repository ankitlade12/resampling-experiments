"""Reproducible run manifests and dataset fingerprints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from configs.experiment import PROTOCOL_VERSION

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "feature-engine",
)


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_fingerprint(values):
    """Hash values plus schema without depending on pickle serialization."""
    digest = hashlib.sha256()
    if isinstance(values, (pd.DataFrame, pd.Series)):
        schema = {
            "type": type(values).__name__,
            "shape": values.shape,
            "columns": list(values.columns) if hasattr(values, "columns") else None,
            "dtypes": [str(dtype) for dtype in np.atleast_1d(values.dtypes)],
        }
        digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes()
        )
    else:
        array = np.asarray(values)
        schema = {"type": "ndarray", "shape": array.shape, "dtype": str(array.dtype)}
        digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
        if array.dtype.hasobject:
            digest.update(json.dumps(array.tolist(), default=str).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def fingerprint_dataset(X_train, y_train, X_test, y_test, *, groups=None):
    """Return split-specific hashes so changed data cannot reuse old models."""
    result = {
        "X_train": array_fingerprint(X_train),
        "y_train": array_fingerprint(y_train),
        "X_test": array_fingerprint(X_test),
        "y_test": array_fingerprint(y_test),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    if groups is not None:
        result["test_groups"] = array_fingerprint(groups)
    result["fingerprint"] = _json_hash(result)
    return result


def current_code_state(repo_root):
    """Capture the exact Git commit and reject invisible uncommitted code."""
    repo_root = Path(repo_root)

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _environment():
    packages = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def initialize_run(
    output_dir, config, repo_root, *, code_state=None, allow_dirty=False
):
    """Create or validate a manifest before any resumable artifacts are used."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    code = current_code_state(repo_root) if code_state is None else code_state
    if code["dirty"] and not allow_dirty:
        raise RuntimeError("Refusing a research run with uncommitted tracked changes.")

    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "config": config,
        "config_fingerprint": _json_hash(config),
        "code": code,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ("protocol_version", "config_fingerprint", "code"):
            if existing.get(key) != identity[key]:
                raise RuntimeError(
                    f"Run manifest mismatch for {key}; use a new output directory."
                )
        return manifest_path

    legacy_items = [path.name for path in output_dir.iterdir()]
    if legacy_items:
        raise RuntimeError(
            "Output directory contains artifacts without a run manifest; "
            "use a new directory rather than mixing protocol versions."
        )

    manifest = {
        **identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "datasets": {},
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def register_dataset(manifest_path, dataset, fingerprint):
    """Add a dataset split hash, or verify it exactly when resuming."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    existing = manifest["datasets"].get(dataset)
    if existing is not None and existing != fingerprint:
        raise RuntimeError(
            f"Dataset fingerprint changed for {dataset}; use a new output directory."
        )
    manifest["datasets"][dataset] = fingerprint
    _write_json_atomic(manifest_path, manifest)
