"""Issue #3647 — LocalSandbox must compile its path-rewrite regexes once per
sandbox (cached), not on every bash/read_file/write_file call, while keeping
the exact same rewriting behavior.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping


def _make_sandbox(tmp_path: Path) -> LocalSandbox:
    ws = tmp_path / "workspace"
    skills = tmp_path / "skills"
    ws.mkdir()
    skills.mkdir()
    return LocalSandbox(
        id="test",
        path_mappings=[
            PathMapping(container_path="/mnt/user-data/workspace", local_path=str(ws)),
            PathMapping(container_path="/mnt/skills", local_path=str(skills), read_only=True),
        ],
    )


def test_patterns_are_compiled_once_and_cached(tmp_path):
    sb = _make_sandbox(tmp_path)
    # Each cached_property returns the identical object across accesses.
    assert sb._command_pattern is sb._command_pattern
    assert sb._content_pattern is sb._content_pattern
    assert sb._reverse_output_patterns is sb._reverse_output_patterns
    # Two mappings -> two reverse-output patterns.
    assert len(sb._reverse_output_patterns) == 2


def test_empty_mappings_yield_no_pattern(tmp_path):
    sb = LocalSandbox(id="empty", path_mappings=[])
    assert sb._command_pattern is None
    assert sb._content_pattern is None
    assert sb._reverse_output_patterns == []
    # No mappings -> command/content pass through unchanged.
    assert sb._resolve_paths_in_command("echo hello") == "echo hello"
    assert sb._resolve_paths_in_content("plain text") == "plain text"


def test_command_paths_resolved_to_local(tmp_path):
    sb = _make_sandbox(tmp_path)
    ws_local = str((tmp_path / "workspace").resolve())
    out = sb._resolve_paths_in_command("cat /mnt/user-data/workspace/foo.txt")
    assert out == f"cat {ws_local}/foo.txt"
    # Calling again uses the cached pattern and produces the same result.
    assert sb._resolve_paths_in_command("cat /mnt/user-data/workspace/foo.txt") == out


def test_segment_boundary_not_matched_inside_longer_name(tmp_path):
    sb = _make_sandbox(tmp_path)
    # "/mnt/skills-extra" must NOT be rewritten by the "/mnt/skills" mapping.
    out = sb._resolve_paths_in_command("ls /mnt/skills-extra/data")
    assert out == "ls /mnt/skills-extra/data"


def test_reverse_resolve_output_maps_local_back_to_container(tmp_path):
    sb = _make_sandbox(tmp_path)
    ws_local = str((tmp_path / "workspace").resolve())
    out = sb._reverse_resolve_paths_in_output(f"wrote {ws_local}/foo.txt ok")
    assert out == "wrote /mnt/user-data/workspace/foo.txt ok"


def test_resolved_paths_and_sorted_views_are_cached(tmp_path):
    sb = _make_sandbox(tmp_path)
    # Resolved-local map and sorted views are computed once and reused.
    assert sb._resolved_local_paths is sb._resolved_local_paths
    assert sb._mappings_by_container_specificity is sb._mappings_by_container_specificity
    assert sb._mappings_by_local_specificity is sb._mappings_by_local_specificity
    # Map covers every mapping with its filesystem-resolved local root.
    assert set(sb._resolved_local_paths.values()) == {
        str((tmp_path / "workspace").resolve()),
        str((tmp_path / "skills").resolve()),
    }
    # Most-specific (longest) container path is ordered first.
    assert sb._mappings_by_container_specificity[0].container_path == "/mnt/user-data/workspace"


def test_forward_resolution_behavior_unchanged(tmp_path):
    sb = _make_sandbox(tmp_path)
    ws_local = str((tmp_path / "workspace").resolve())
    # Container path resolves to the mapped local path.
    assert sb._resolve_path("/mnt/user-data/workspace/sub/foo.txt") == f"{ws_local}/sub/foo.txt"
    # An unmapped path is returned unchanged.
    assert sb._resolve_path("/etc/hosts") == "/etc/hosts"


def test_read_only_mount_detected(tmp_path):
    sb = _make_sandbox(tmp_path)
    skills_local = str((tmp_path / "skills").resolve())
    ws_local = str((tmp_path / "workspace").resolve())
    assert sb._is_read_only_path(f"{skills_local}/a.md") is True
    assert sb._is_read_only_path(f"{ws_local}/a.txt") is False


# ---------------------------------------------------------------------------
# Windows / Git Bash (MSYS) path emission
# ---------------------------------------------------------------------------


def test_command_posix_paths_normalizes_backslashes(tmp_path):
    """With posix_paths=True the resolved host path uses forward slashes.

    On Windows the native resolution yields a backslash path (e.g. ``D:\\a\\b``);
    Git Bash `sh` eats those backslashes as escapes and destroys the path. The
    posix variant must contain no backslashes and equal the native output with
    `\\` -> `/`. On POSIX hosts there are no backslashes, so both are identical.
    """
    sb = _make_sandbox(tmp_path)
    cmd = "cd /mnt/user-data/workspace && unzip -o demo.zip"

    native = sb._resolve_paths_in_command(cmd)
    posix = sb._resolve_paths_in_command(cmd, posix_paths=True)

    assert "\\" not in posix
    assert posix == native.replace("\\", "/")
    # The virtual prefix is actually resolved (not passed through verbatim).
    assert "/mnt/user-data/workspace" not in posix


def test_execute_command_uses_posix_paths_for_msys_shell(tmp_path):
    """execute_command must resolve paths in forward-slash form when the selected
    shell is Git Bash/MSYS, so `cd /mnt/...` survives instead of being mangled."""
    sb = _make_sandbox(tmp_path)
    fake_msys_sh = "C:/Program Files/Git/usr/bin/sh.exe"

    with (
        mock.patch.object(sb, "_get_shell", return_value=fake_msys_sh),
        mock.patch(
            "deerflow.sandbox.local.local_sandbox.subprocess.run",
            return_value=SimpleNamespace(stdout="ok", stderr="", returncode=0),
        ) as run,
    ):
        sb.execute_command("cd /mnt/user-data/workspace && unzip -o demo.zip")

    # The command string handed to the shell (last positional arg) must carry a
    # forward-slash host path and no backslashes, and must not still contain the
    # unresolved virtual prefix.
    sent_args = run.call_args.args[0]
    command_str = sent_args[-1]
    assert "\\" not in command_str
    assert "/mnt/user-data/workspace" not in command_str
