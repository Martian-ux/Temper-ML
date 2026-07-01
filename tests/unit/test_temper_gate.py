from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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

    monkeypatch.setattr(gate.shutil, "which", lambda name: "uv" if name == "uv" else None)
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


def test_maintenance_runs_compile_and_unit_checks_through_uv(monkeypatch):
    gate = load_gate_module()
    calls = []

    monkeypatch.setattr(gate.shutil, "which", lambda name: "uv" if name == "uv" else None)
    monkeypatch.setattr(gate.subprocess, "run", successful_run(calls))

    assert gate.main(["maintenance"]) == 0

    assert [call["command"] for call in calls] == [
        ["uv", "run", "python", "-m", "compileall", "-q", "src"],
        ["uv", "run", "pytest", "tests/unit"],
    ]


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
    monkeypatch.setattr(gate.tempfile, "TemporaryDirectory", lambda prefix: FakeTemporaryDirectory())
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


def test_fixture_help_does_not_spawn_bash(monkeypatch, capsys):
    gate = load_gate_module()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fixture help should be rendered by Python")

    monkeypatch.setattr(gate.subprocess, "run", fail_if_called)

    assert gate.main(["fixture-help"]) == 0

    output = capsys.readouterr().out
    assert "Usage:" in output
    assert "fixture walkthrough" in output
