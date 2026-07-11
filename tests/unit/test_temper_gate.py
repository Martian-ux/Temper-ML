from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def load_gate_module():
    gate_path = Path(__file__).resolve().parents[2] / "scripts" / "temper-gate.py"
    spec = importlib.util.spec_from_file_location("temper_gate", gate_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def successful_run(calls):
    def run(command, *, cwd=None, env=None):
        calls.append({"command": command, "cwd": cwd, "env": env})
        return subprocess.CompletedProcess(command, 0)

    return run


def test_unit_runs_pytest_through_uv(monkeypatch, capsys):
    gate = load_gate_module()
    calls = []

    monkeypatch.setattr(
        gate.shutil, "which", lambda name: "uv" if name == "uv" else None
    )
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.main(["unit"]) == 0

    assert calls == [
        {
            "command": ["uv", "run", "pytest", "tests/unit"],
            "cwd": gate.REPO_ROOT,
            "env": None,
        }
    ]
    assert "+ uv run pytest tests/unit" in capsys.readouterr().out


def test_setup_requires_the_committed_lockfile(monkeypatch):
    gate = load_gate_module()
    calls = []

    monkeypatch.setattr(
        gate.shutil, "which", lambda name: "uv" if name == "uv" else None
    )
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.main(["setup"]) == 0
    assert [call["command"] for call in calls] == [["uv", "sync", "--dev", "--locked"]]


def test_maintenance_runs_compile_and_complete_suite_through_uv(monkeypatch):
    gate = load_gate_module()
    calls = []

    monkeypatch.setattr(
        gate.shutil, "which", lambda name: "uv" if name == "uv" else None
    )
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.main(["maintenance"]) == 0

    assert [call["command"] for call in calls] == [
        ["uv", "run", "ruff", "format", "--check", "src", "tests", "scripts"],
        ["uv", "run", "ruff", "check", "src", "tests", "scripts"],
        ["uv", "run", "mypy", "src"],
        ["uv", "run", "python", "-m", "compileall", "-q", "src"],
        ["uv", "run", "pytest", "tests"],
    ]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "format",
            ["uv", "run", "ruff", "format", "--check", "src", "tests", "scripts"],
        ),
        ("lint", ["uv", "run", "ruff", "check", "src", "tests", "scripts"]),
        ("typecheck", ["uv", "run", "mypy", "src"]),
        ("compile", ["uv", "run", "python", "-m", "compileall", "-q", "src"]),
    ],
)
def test_quality_commands_run_through_uv(monkeypatch, command, expected):
    gate = load_gate_module()
    calls = []

    monkeypatch.setattr(
        gate.shutil, "which", lambda name: "uv" if name == "uv" else None
    )
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.main([command]) == 0

    assert [call["command"] for call in calls] == [expected]


def test_diff_hygiene_runs_without_resolving_uv(monkeypatch):
    gate = load_gate_module()
    calls = []

    monkeypatch.setattr(
        gate.shutil, "which", lambda name: "git" if name == "git" else None
    )
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.main(["diff"]) == 0
    assert [call["command"] for call in calls] == [["git", "diff", "--check"]]


def test_missing_uv_requires_explicit_bootstrap(monkeypatch, capsys):
    gate = load_gate_module()
    calls = []

    monkeypatch.setattr(gate.shutil, "which", lambda name: None)
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.main(["unit"]) == 127

    assert calls == []
    assert "--bootstrap-uv temp" in capsys.readouterr().err


def test_explicit_temp_bootstrap_uses_python_module_outside_repo(monkeypatch, tmp_path):
    gate = load_gate_module()
    calls = []
    bootstrap_dir = tmp_path / "temper-uv-bootstrap"

    class FakeTemporaryDirectory:
        def __enter__(self):
            return str(bootstrap_dir)

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(gate.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        gate.tempfile, "TemporaryDirectory", lambda prefix: FakeTemporaryDirectory()
    )
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.main(["--bootstrap-uv", "temp", "unit"]) == 0

    assert calls[0]["command"] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(bootstrap_dir),
        "uv",
    ]
    assert calls[1]["command"] == [
        sys.executable,
        "-m",
        "uv",
        "run",
        "pytest",
        "tests/unit",
    ]
    assert str(bootstrap_dir) in calls[1]["env"]["PYTHONPATH"]


def test_temp_bootstrap_install_failure_propagates_and_skips_uv_gates(
    monkeypatch, tmp_path
):
    gate = load_gate_module()
    calls = []
    bootstrap_dir = tmp_path / "temper-uv-bootstrap"

    class FakeTemporaryDirectory:
        def __enter__(self):
            return str(bootstrap_dir)

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fail_bootstrap(command, *, cwd=None, env=None):
        calls.append({"command": command, "cwd": cwd, "env": env})
        return subprocess.CompletedProcess(command, 23)

    monkeypatch.setattr(gate.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        gate.tempfile, "TemporaryDirectory", lambda prefix: FakeTemporaryDirectory()
    )
    monkeypatch.setattr(gate.subprocess, "run", fail_bootstrap)

    assert gate.main(["--bootstrap-uv", "temp", "all"]) == 23
    assert [call["command"] for call in calls] == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            str(bootstrap_dir),
            "uv",
        ]
    ]


def test_all_runs_dependency_checks_once_through_resolved_uv(monkeypatch):
    gate = load_gate_module()
    calls = []
    uv_command = [sys.executable, "-m", "uv"]
    uv_env = {"PYTHONPATH": "synthetic-bootstrap-path"}

    monkeypatch.setattr(
        gate.shutil, "which", lambda name: "git" if name == "git" else None
    )
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.run_all(uv_command, uv_env) == 0

    assert [call["command"] for call in calls] == [
        [*uv_command, "sync", "--dev", "--locked"],
        [*uv_command, "run", "ruff", "format", "--check", "src", "tests", "scripts"],
        [*uv_command, "run", "ruff", "check", "src", "tests", "scripts"],
        [*uv_command, "run", "mypy", "src"],
        [*uv_command, "run", "python", "-m", "compileall", "-q", "src"],
        [*uv_command, "run", "pytest", "tests"],
        ["git", "diff", "--check"],
    ]
    assert [call["env"] for call in calls[:6]] == [uv_env] * 6


def test_all_propagates_uv_failure_and_short_circuits_later_gates(monkeypatch):
    gate = load_gate_module()
    calls = []

    def fail_maintenance_compile(command, *, cwd=None, env=None):
        calls.append({"command": command, "cwd": cwd, "env": env})
        status = 41 if command[1:4] == ["run", "python", "-m"] else 0
        return subprocess.CompletedProcess(command, status)

    monkeypatch.setattr(gate.shutil, "which", lambda name: name)
    monkeypatch.setattr(gate.subprocess, "run", fail_maintenance_compile)

    assert gate.main(["all"]) == 41
    assert [call["command"] for call in calls] == [
        ["uv", "sync", "--dev", "--locked"],
        ["uv", "run", "ruff", "format", "--check", "src", "tests", "scripts"],
        ["uv", "run", "ruff", "check", "src", "tests", "scripts"],
        ["uv", "run", "mypy", "src"],
        ["uv", "run", "python", "-m", "compileall", "-q", "src"],
    ]


def test_all_runs_diff_hygiene_only_after_earlier_gates_succeed(monkeypatch):
    gate = load_gate_module()
    events = []

    monkeypatch.setattr(
        gate,
        "run_setup",
        lambda uv_command, uv_env: events.append("setup") or 0,
    )
    monkeypatch.setattr(
        gate,
        "run_maintenance",
        lambda uv_command, uv_env: events.append("maintenance") or 0,
    )
    monkeypatch.setattr(
        gate, "run_fixture_help", lambda: events.append("fixture-help") or 0
    )
    monkeypatch.setattr(
        gate, "run_diff_hygiene", lambda: events.append("diff-hygiene") or 0
    )

    assert gate.run_all(["uv"], None) == 0
    assert events == [
        "setup",
        "maintenance",
        "fixture-help",
        "diff-hygiene",
    ]


def test_fixture_help_does_not_spawn_bash(monkeypatch, capsys):
    gate = load_gate_module()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fixture help should be rendered by Python")

    monkeypatch.setattr(gate.subprocess, "run", fail_if_called)

    assert gate.main(["fixture-help"]) == 0

    output = capsys.readouterr().out
    assert "Usage:" in output
    assert "fixture walkthrough" in output


def test_ci_gate_runs_on_ubuntu_and_windows() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "temper-gate.yml"
    ).read_text(encoding="utf-8")

    assert "os: [ubuntu-latest, windows-latest]" in workflow
    assert "runs-on: ${{ matrix.os }}" in workflow
    assert "python scripts/temper-gate.py all" in workflow
