#!/usr/bin/env python3
"""Cross-platform operational gates for Temper ML."""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_HELP = """Usage: python scripts/temper-gate.py fixture-help
Usage: bash scripts/temper-fixture-walkthrough.sh --help

TML-001 provides this command target for the future deterministic fixture
walkthrough. The walkthrough execution path is intentionally deferred until the
fixture skeleton stage.

This is fixture walkthrough help only; execution arrives in the fixture
skeleton stage.
"""


class GateExit(Exception):
    """Internal exception for returning a specific process status."""

    def __init__(self, code: int):
        self.code = code


def _format_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> int:
    print(f"+ {_format_command(command)}", flush=True)
    completed = subprocess.run(list(command), cwd=cwd, env=env)
    return completed.returncode


@contextlib.contextmanager
def resolve_uv(
    bootstrap_mode: str,
) -> Iterator[tuple[list[str], dict[str, str] | None]]:
    if shutil.which("uv"):
        yield ["uv"], None
        return

    if bootstrap_mode != "temp":
        print(
            "uv was not found on PATH. Install uv or rerun with "
            "`--bootstrap-uv temp` to install uv into a temporary directory.",
            file=sys.stderr,
        )
        raise GateExit(127)

    with tempfile.TemporaryDirectory(prefix="temper-uv-") as tool_dir:
        install_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            tool_dir,
            "uv",
        ]
        install_status = run_command(install_command)
        if install_status != 0:
            raise GateExit(install_status)

        env = os.environ.copy()
        env["PYTHONPATH"] = tool_dir + os.pathsep + env.get("PYTHONPATH", "")
        yield [sys.executable, "-m", "uv"], env


def run_uv(
    command: Sequence[str], uv_command: Sequence[str], uv_env: dict[str, str] | None
) -> int:
    return run_command([*uv_command, *command], env=uv_env)


def run_setup(uv_command: Sequence[str], uv_env: dict[str, str] | None) -> int:
    return run_uv(["sync", "--dev", "--locked"], uv_command, uv_env)


def run_unit(uv_command: Sequence[str], uv_env: dict[str, str] | None) -> int:
    return run_uv(["run", "pytest", "tests/unit"], uv_command, uv_env)


def run_format(uv_command: Sequence[str], uv_env: dict[str, str] | None) -> int:
    return run_uv(
        ["run", "ruff", "format", "--check", "src", "tests", "scripts"],
        uv_command,
        uv_env,
    )


def run_lint(uv_command: Sequence[str], uv_env: dict[str, str] | None) -> int:
    return run_uv(
        ["run", "ruff", "check", "src", "tests", "scripts"],
        uv_command,
        uv_env,
    )


def run_typecheck(uv_command: Sequence[str], uv_env: dict[str, str] | None) -> int:
    return run_uv(["run", "mypy", "src"], uv_command, uv_env)


def run_compile(uv_command: Sequence[str], uv_env: dict[str, str] | None) -> int:
    return run_uv(
        ["run", "python", "-m", "compileall", "-q", "src"], uv_command, uv_env
    )


def run_maintenance(uv_command: Sequence[str], uv_env: dict[str, str] | None) -> int:
    for gate in (
        run_format,
        run_lint,
        run_typecheck,
        run_compile,
        run_unit,
    ):
        status = gate(uv_command, uv_env)
        if status != 0:
            return status
    return 0


def run_fixture_help() -> int:
    print(FIXTURE_HELP, end="")
    return 0


def run_diff_hygiene() -> int:
    if not shutil.which("git"):
        print("Skipping diff hygiene: git was not found on PATH.", file=sys.stderr)
        return 0
    return run_command(["git", "diff", "--check"])


def run_all(uv_command: Sequence[str], uv_env: dict[str, str] | None) -> int:
    for gate in (
        lambda: run_setup(uv_command, uv_env),
        lambda: run_maintenance(uv_command, uv_env),
        run_fixture_help,
        run_diff_hygiene,
    ):
        status = gate()
        if status != 0:
            return status
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="temper-gate",
        description="Run Temper ML operational gates without requiring Bash.",
    )
    parser.add_argument(
        "--bootstrap-uv",
        choices=["none", "temp"],
        default="none",
        help=(
            "When uv is missing, `temp` installs uv into a temporary directory "
            "for this process only. The default does not install anything."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("setup", help="Install development dependencies through uv.")
    subparsers.add_parser(
        "maintenance", help="Run static, compile, and unit checks through uv."
    )
    subparsers.add_parser("unit", help="Run unit tests through uv.")
    subparsers.add_parser("format", help="Check Python formatting through uv.")
    subparsers.add_parser("lint", help="Run Python lint checks through uv.")
    subparsers.add_parser("typecheck", help="Run Python type checks through uv.")
    subparsers.add_parser("compile", help="Compile Python sources through uv.")
    subparsers.add_parser(
        "fixture-help", help="Show fixture walkthrough help without Bash."
    )
    subparsers.add_parser(
        "diff", help="Check the working diff for whitespace errors without uv."
    )
    subparsers.add_parser(
        "all", help="Run setup, maintenance, fixture help, and diff hygiene."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "fixture-help":
        return run_fixture_help()
    if args.command == "diff":
        return run_diff_hygiene()

    try:
        with resolve_uv(args.bootstrap_uv) as (uv_command, uv_env):
            if args.command == "setup":
                return run_setup(uv_command, uv_env)
            if args.command == "maintenance":
                return run_maintenance(uv_command, uv_env)
            if args.command == "unit":
                return run_unit(uv_command, uv_env)
            if args.command == "format":
                return run_format(uv_command, uv_env)
            if args.command == "lint":
                return run_lint(uv_command, uv_env)
            if args.command == "typecheck":
                return run_typecheck(uv_command, uv_env)
            if args.command == "compile":
                return run_compile(uv_command, uv_env)
            if args.command == "all":
                return run_all(uv_command, uv_env)
    except GateExit as exc:
        return exc.code

    raise AssertionError(f"unhandled gate command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
