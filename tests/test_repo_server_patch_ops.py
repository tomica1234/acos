import hashlib

import pytest

from packages.mcp_client.fake import RepoServer


def test_repo_server_apply_patch_delete_rename_and_integrity(tmp_path) -> None:
    server = RepoServer(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    result = server.apply_patch(
        "source.py",
        operation="rename",
        new_path="package/renamed.py",
        base_sha256=digest,
        expected_old_content="VALUE = 1\n",
    )

    assert result["operation"] == "rename"
    assert not source.exists()
    renamed = tmp_path / "package" / "renamed.py"
    assert renamed.read_text(encoding="utf-8") == "VALUE = 1\n"

    server.apply_patch("package/renamed.py", operation="delete")

    assert not renamed.exists()


def test_repo_server_apply_patch_reports_integrity_mismatch(tmp_path) -> None:
    server = RepoServer(tmp_path)
    (tmp_path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected_old_content mismatch"):
        server.apply_patch(
            "source.py",
            content="VALUE = 2\n",
            expected_old_content="VALUE = 0\n",
        )
