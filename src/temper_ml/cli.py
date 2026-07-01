"""Minimal Temper ML command-line entry point."""

from __future__ import annotations

import argparse

from temper_ml import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="temper")
    parser.add_argument("--version", action="version", version=f"temper-ml {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
