import json

import numpy as np
import pytest

from functions.provenance import (
    array_fingerprint,
    fingerprint_dataset,
    initialize_run,
    register_dataset,
)

CODE = {"commit": "abc123", "dirty": False}


def test_array_fingerprint_changes_with_values():
    assert array_fingerprint(np.array([1, 2])) != array_fingerprint(np.array([1, 3]))


def test_manifest_allows_only_exact_resume(tmp_path):
    manifest = initialize_run(
        tmp_path, {"scoring": "roc_auc"}, tmp_path, code_state=CODE
    )
    assert (
        initialize_run(tmp_path, {"scoring": "roc_auc"}, tmp_path, code_state=CODE)
        == manifest
    )

    with pytest.raises(RuntimeError, match="config_fingerprint"):
        initialize_run(tmp_path, {"scoring": "brier"}, tmp_path, code_state=CODE)


def test_manifest_rejects_legacy_artifacts(tmp_path):
    (tmp_path / "old-model.pkl").write_bytes(b"legacy")
    with pytest.raises(RuntimeError, match="without a run manifest"):
        initialize_run(tmp_path, {}, tmp_path, code_state=CODE)


def test_dataset_fingerprint_is_checked_on_resume(tmp_path):
    manifest = initialize_run(tmp_path, {}, tmp_path, code_state=CODE)
    first = fingerprint_dataset([[1]], [0], [[2]], [1])
    register_dataset(manifest, "demo", first)
    register_dataset(manifest, "demo", first)

    changed = fingerprint_dataset([[9]], [0], [[2]], [1])
    with pytest.raises(RuntimeError, match="Dataset fingerprint changed"):
        register_dataset(manifest, "demo", changed)

    assert json.loads(manifest.read_text())["datasets"]["demo"] == first


def test_dirty_code_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="uncommitted"):
        initialize_run(
            tmp_path,
            {},
            tmp_path,
            code_state={"commit": "abc123", "dirty": True},
        )
