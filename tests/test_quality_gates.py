import pytest

from packages.orchestrator.quality_gates import (
    QualityGateError,
    artifact_path_exists,
    ensure_required_artifacts_exist,
    ensure_test_patch_quality,
    invalid_artifact_paths,
    valid_artifact_paths,
)
from packages.schemas.agent_outputs import FilePatch


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


def test_test_patch_quality_rejects_frontend_skipped_tests() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { describe, it } from 'vitest'\n\n"
            "describe('project scaffold', () => {\n"
            "  it.skip('loads', () => {})\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_frontend_todo_tests() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { test } from 'vitest'\n\n"
            "test.todo('loads project scaffold')\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_frontend_skip_each_tests() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test.skip.each([['project scaffold']])('loads %s', (label) => {\n"
            "  expect(label).toContain('project')\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_frontend_only_each_tests() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { describe, expect, it } from 'vitest'\n\n"
            "describe.only.each([['project scaffold']])('suite %s', (label) => {\n"
            "  it('loads', () => {\n"
            "    expect(label).toContain('project')\n"
            "  })\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_vacuous_frontend_assertions() -> None:
    patch = FilePatch(
        path="src/App.spec.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('placeholder', () => {\n"
            "  expect(true).toBe(true)\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_vacuous_frontend_literal_expectations() -> None:
    patch = FilePatch(
        path="src/App.spec.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('placeholder', () => {\n"
            "  expect('project scaffold').toContain('scaffold')\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_vacuous_frontend_expectation_update_diff() -> None:
    patch = FilePatch(
        path="src/App.spec.tsx",
        operation="update",
        unified_diff=(
            "--- src/App.spec.tsx\n"
            "+++ src/App.spec.tsx\n"
            "@@ -2,5 +2,5 @@\n"
            " test('loads app', () => {\n"
            "-  expect(screen.getByText('Ready')).toBeInTheDocument()\n"
            "+  expect(2).toEqual(2)\n"
            " })\n"
        ),
    )

    with pytest.raises(QualityGateError, match="fixer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="fixer")


def test_test_patch_quality_rejects_vacuous_frontend_negated_expectations() -> None:
    patch = FilePatch(
        path="src/App.spec.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('placeholder', () => {\n"
            "  expect(true).not.toBe(false)\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_vacuous_frontend_comparison_expectations() -> None:
    patch = FilePatch(
        path="src/App.spec.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('placeholder', () => {\n"
            "  expect(3).toBeGreaterThan(1)\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_vacuous_frontend_truthiness_expectations() -> None:
    patch = FilePatch(
        path="src/App.spec.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('placeholder', () => {\n"
            "  expect('ready').toBeTruthy()\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_allows_frontend_variable_expectation() -> None:
    patch = FilePatch(
        path="src/App.spec.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('uses app output', () => {\n"
            "  const label = renderAppLabel()\n"
            "  expect(label).toContain('project scaffold')\n"
            "})\n"
        ),
    )

    ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_empty_frontend_tests() -> None:
    patch = FilePatch(
        path="backend/test/scaffold.test.js",
        operation="create",
        content=(
            "describe('scaffold backend', () => {\n"
            "  it('exists', () => {})\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_frontend_test_without_assertion() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { test } from 'vitest'\n\n"
            "test('loads project scaffold', () => {\n"
            "  renderProjectScaffold()\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_frontend_comment_only_expectation() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { test } from 'vitest'\n\n"
            "test('loads project scaffold', () => {\n"
            "  // expect(renderProjectScaffold()).toContain('project')\n"
            "  renderProjectScaffold()\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_frontend_inline_comment_only_expectation() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { test } from 'vitest'\n\n"
            "test('loads project scaffold', () => {\n"
            "  renderProjectScaffold() // expect(label).toContain('project')\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_one_frontend_test_without_assertion() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('checks scaffold label', () => {\n"
            "  const label = renderProjectScaffold()\n"
            "  expect(label).toContain('project')\n"
            "})\n\n"
            "test('loads project scaffold', () => {\n"
            "  renderProjectScaffold()\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_frontend_function_test_without_assertion() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('checks scaffold label', () => {\n"
            "  const label = renderProjectScaffold()\n"
            "  expect(label).toContain('project')\n"
            "})\n\n"
            "test('loads project scaffold', function () {\n"
            "  renderProjectScaffold()\n"
            "})\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_allows_frontend_function_test_with_expectation() -> None:
    patch = FilePatch(
        path="frontend/test/project_scaffold.test.tsx",
        operation="create",
        content=(
            "import { expect, test } from 'vitest'\n\n"
            "test('loads project scaffold', function () {\n"
            "  const label = renderProjectScaffold()\n"
            "  expect(label).toContain('project')\n"
            "})\n"
        ),
    )

    ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_empty_test_file_create() -> None:
    patch = FilePatch(
        path="tests/test_project_setup.py",
        operation="create",
        content="",
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_removal_only_test_diff() -> None:
    patch = FilePatch(
        path="tests/test_project_setup.py",
        operation="update",
        unified_diff=(
            "--- tests/test_project_setup.py\n"
            "+++ tests/test_project_setup.py\n"
            "@@ -1,2 +0,0 @@\n"
            "-def test_project_setup() -> None:\n"
            "-    assert 'project' in 'project setup'\n"
        ),
    )

    with pytest.raises(QualityGateError, match="fixer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="fixer")


def test_test_patch_quality_rejects_assertion_removal_with_non_assertion_addition() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="update",
        unified_diff=(
            "--- tests/test_feature.py\n"
            "+++ tests/test_feature.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def test_feature() -> None:\n"
            "-    assert VALUE == 1\n"
            "+    # Covered by manual validation.\n"
        ),
    )

    with pytest.raises(QualityGateError, match="fixer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="fixer")


def test_test_patch_quality_allows_assertion_replacement_diff() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="update",
        unified_diff=(
            "--- tests/test_feature.py\n"
            "+++ tests/test_feature.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def test_feature() -> None:\n"
            "-    assert VALUE == 1\n"
            "+    assert VALUE == 2\n"
        ),
    )

    ensure_test_patch_quality([patch], role="fixer")


def test_test_patch_quality_rejects_vacuous_python_literal_assertions() -> None:
    patch = FilePatch(
        path="tests/test_project_setup.py",
        operation="create",
        content=(
            "def test_project_setup_placeholder() -> None:\n"
            "    assert 1 == 1\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_vacuous_python_chained_comparison() -> None:
    patch = FilePatch(
        path="tests/test_project_setup.py",
        operation="create",
        content=(
            "def test_project_setup_placeholder() -> None:\n"
            "    assert 1 < 2 < 3\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_allows_python_variable_chained_comparison() -> None:
    patch = FilePatch(
        path="tests/test_project_setup.py",
        operation="create",
        content=(
            "def test_project_setup_order() -> None:\n"
            "    first, second, third = compute_project_order()\n"
            "    assert first < second < third\n"
        ),
    )

    ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_vacuous_assertion_replacement_diff() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="update",
        unified_diff=(
            "--- tests/test_feature.py\n"
            "+++ tests/test_feature.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def test_feature() -> None:\n"
            "-    assert result.status_code == 200\n"
            "+    assert 'project' in 'project setup'\n"
        ),
    )

    with pytest.raises(QualityGateError, match="fixer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="fixer")


def test_test_patch_quality_rejects_python_pass_tests() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="create",
        content="def test_placeholder() -> None:\n    pass\n",
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_python_test_without_assertion() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="create",
        content="def test_feature() -> None:\n    build_feature()\n",
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_python_comment_only_assertion() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="create",
        content=(
            "def test_feature() -> None:\n"
            "    # assert build_feature().status == 'ready'\n"
            "    build_feature()\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_python_inline_comment_only_assertion() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="create",
        content=(
            "def test_feature() -> None:\n"
            "    build_feature()  # assert build_feature().status == 'ready'\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_allows_assertions_with_trailing_comments() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="create",
        content=(
            "def test_feature() -> None:\n"
            "    assert build_feature().status == 'ready'  # verifies status\n"
        ),
    )

    ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_one_python_test_without_assertion() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="create",
        content=(
            "def test_feature_status() -> None:\n"
            "    result = build_feature()\n"
            "    assert result.status == 'ready'\n\n"
            "def test_feature_side_effect() -> None:\n"
            "    build_feature()\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_allows_test_helper_without_new_test_case() -> None:
    patch = FilePatch(
        path="tests/helpers.py",
        operation="update",
        content="def build_feature_payload() -> dict[str, str]:\n    return {'name': 'feature'}\n",
    )

    ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_python_importorskip_tests() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="create",
        content=(
            "import pytest\n\n"
            "pytest.importorskip('missing_optional_dependency')\n\n"
            "def test_feature() -> None:\n"
            "    assert feature_enabled()\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_unittest_skip_tests() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="create",
        content=(
            "import unittest\n\n"
            "@unittest.skip('not ready')\n"
            "def test_feature() -> None:\n"
            "    assert feature_enabled()\n"
        ),
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_rejects_test_file_deletes() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="delete",
    )

    with pytest.raises(QualityGateError, match="fixer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="fixer")


def test_test_patch_quality_rejects_renaming_tests_outside_test_tree() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="rename",
        new_path="docs/test_feature_backup.py",
    )

    with pytest.raises(QualityGateError, match="test_writer attempted to weaken tests"):
        ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_allows_test_rename_inside_test_tree() -> None:
    patch = FilePatch(
        path="tests/test_feature.py",
        operation="rename",
        new_path="tests/test_feature_smoke.py",
    )

    ensure_test_patch_quality([patch], role="test_writer")


def test_test_patch_quality_allows_non_test_files_with_skip_text() -> None:
    patch = FilePatch(
        path="frontend/src/App.tsx",
        operation="create",
        content="export const label = 'skip(optional setup)'\n",
    )

    ensure_test_patch_quality([patch], role="test_writer")
