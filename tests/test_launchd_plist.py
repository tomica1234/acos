from __future__ import annotations

import plistlib

from apps.cli import build_launchd_plist


def test_build_launchd_plist_contains_expected_runtime_fields(tmp_path) -> None:
    payload = build_launchd_plist(workspace_root=tmp_path, config_dir="configs")
    plist = plistlib.loads(plistlib.dumps(payload))

    assert plist["Label"] == "com.acos.worker"
    assert plist["ProgramArguments"][:4] == ["acos", "worker", "run", "--forever"]
    assert plist["WorkingDirectory"] == str(tmp_path.resolve())
    assert plist["KeepAlive"] is True
    assert plist["RunAtLoad"] is True
    assert plist["StandardOutPath"].endswith(".acos/logs/worker.out.log")
    assert plist["StandardErrorPath"].endswith(".acos/logs/worker.err.log")
