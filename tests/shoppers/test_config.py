"""Tests for src.shoppers.config helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.shoppers import config


def test_load_params(tmp_project: Path) -> None:
    params = config.load_params()
    assert params["prepare"]["target"] == "Revenue"
    assert params["seed"] == 42


def test_load_params_rejects_non_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    monkeypatch.setattr(config, "PARAMS_PATH", bad)
    with pytest.raises(ValueError, match="mapping"):
        config.load_params()


def test_flatten_nested() -> None:
    flat = config.flatten({"train": {"seed": 1, "nested": {"x": 2}}, "a": 3})
    assert flat == {"train.seed": 1, "train.nested.x": 2, "a": 3}


def test_write_json_and_ensure_dirs(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "out.json"
    config.write_json(path, {"b": 2, "a": 1})
    text = path.read_text(encoding="utf-8")
    assert '"a": 1' in text
    assert text.endswith("\n")


def test_get_logger() -> None:
    logger = config.get_logger("shoppers.test")
    assert logger.name == "shoppers.test"


def test_lineage_tags_without_git_dvc(tmp_project: Path) -> None:
    tags = config.lineage_tags()
    assert "git_sha" in tags
    assert tags["dvc_data_hash"] == "unknown"
    assert tags["pipeline"] == "dvc"


def test_dvc_data_hash_from_lock(tmp_project: Path) -> None:
    lock = {
        "stages": {
            "prepare": {
                "deps": [
                    {"path": "data/raw/shoppers.csv", "md5": "abc123"},
                ]
            }
        }
    }
    config.DVC_LOCK_PATH.write_text(yaml.safe_dump(lock), encoding="utf-8")
    assert config.dvc_data_hash() == "abc123"


def test_configure_mlflow(tmp_project: Path) -> None:
    config.configure_mlflow()
    import mlflow

    assert mlflow.get_tracking_uri().startswith("file://")


def test_git_helpers(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(cmd: list[str], **_kwargs: object) -> Result:
        if cmd[:2] == ["git", "rev-parse"]:
            return Result("abc123\n")
        if cmd[:2] == ["git", "status"]:
            return Result("")
        raise AssertionError(cmd)

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    assert config.git_sha() == "abc123"
    assert config.git_dirty() is False


def test_git_helpers_dirty(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(cmd: list[str], **_kwargs: object) -> Result:
        if cmd[:2] == ["git", "rev-parse"]:
            return Result("abc123\n")
        return Result(" M file")

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    assert config.git_sha().endswith("-dirty")
    assert config.git_dirty() is True


def test_dvc_data_hash_bad_lock(tmp_project: Path) -> None:
    config.DVC_LOCK_PATH.write_text("- not a mapping\n", encoding="utf-8")
    assert config.dvc_data_hash() == "unknown"
