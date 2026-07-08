import pytest

from packages.orchestrator.quality_gates import (
    QualityGateError,
    artifact_path_exists,
    ensure_required_artifacts_exist,
    invalid_artifact_paths,
    valid_artifact_paths,
)


def test_required_artifacts_reject_windows_absolute_paths(tmp_path) -> None:
    with pytest.raises(QualityGateError, match="invalid target_files"):
        ensure_required_artifacts_exist(
            ["C:\\outside.py"],
            workspace_root=tmp_path,
        )


def test_required_artifacts_reject_directory_like_paths(tmp_path) -> None:
    with pytest.raises(QualityGateError, match="invalid target_files"):
        ensure_required_artifacts_exist(
            ["frontend/src/"],
            workspace_root=tmp_path,
        )


def test_artifact_path_helpers_validate_cross_platform_paths() -> None:
    paths = ["feature.py", "tests/test_feature.py", "../outside.py", "C:\\outside.py"]

    assert valid_artifact_paths(paths) == {"feature.py", "tests/test_feature.py"}
    assert invalid_artifact_paths(paths) == ["../outside.py", "C:\\outside.py"]


def test_artifact_path_exists_requires_file_inside_workspace(tmp_path) -> None:
    (tmp_path / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()

    assert artifact_path_exists("feature.py", workspace_root=tmp_path)
    assert not artifact_path_exists("missing.py", workspace_root=tmp_path)
    assert not artifact_path_exists("docs", workspace_root=tmp_path)
    assert not artifact_path_exists("../outside.py", workspace_root=tmp_path)
    assert not artifact_path_exists("C:\\outside.py", workspace_root=tmp_path)
