"""Public-safe command-line access to a local Temper evidence store."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from typing import NoReturn

from temper_ml import __version__
from temper_ml.domain.projections import ContentIdentity, ProjectionError
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.redaction import PublicSafetyError


class _JsonArgumentParser(argparse.ArgumentParser):
    """Argparse variant whose failures cannot disclose user-supplied values."""

    def error(self, message: str) -> NoReturn:
        del message
        _emit_json(sys.stderr, {"status": "error", "code": "usage_error"})
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(prog="temper")
    parser.add_argument(
        "--version", action="version", version=f"temper-ml {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "verify", "dump"):
        command = commands.add_parser(name)
        command.add_argument("project")
    manifest = commands.add_parser("manifest")
    manifest.add_argument("project")
    manifest.add_argument("--type", dest="record_type", required=True)
    manifest.add_argument("--id", dest="logical_id", required=True)
    manifest.add_argument("--identity")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = _run(arguments)
        encoded = dumps_canonical_json(result)
    except PublicSafetyError as exc:
        _emit_error(exc.code)
        return 3
    except EvidenceError as exc:
        _emit_error(exc.code)
        return 3 if exc.code.startswith("admission_") else 1
    except (OSError, UnicodeError):
        _emit_error("filesystem_error")
        return 4
    except Exception:
        _emit_error("internal_error")
        return 4
    try:
        _write_bytes(sys.stdout, encoded)
    except (OSError, UnicodeError):
        _emit_error("filesystem_error")
        return 4
    return 0


def _run(arguments: argparse.Namespace) -> object:
    store = TypedEvidenceStore(arguments.project)
    if arguments.command in {"status", "verify"}:
        result = store.verify().to_dict()
        result["schema_version"] = "v1"
        result["command"] = arguments.command
        return result
    if arguments.command == "dump":
        return store.public_dump().value
    if arguments.command == "manifest":
        identity = (
            _parse_identity(arguments.identity)
            if arguments.identity is not None
            else None
        )
        return store.inspect_manifest(
            arguments.record_type,
            arguments.logical_id,
            identity,
        ).to_dict()
    raise EvidenceError("unknown_command")


def _parse_identity(value: str) -> ContentIdentity:
    digest = value.removeprefix("sha256:")
    try:
        return ContentIdentity("sha256", digest)
    except ProjectionError:
        raise EvidenceError("invalid_identity") from None


def _emit_error(code: str) -> None:
    _emit_json(sys.stderr, {"status": "error", "code": code})


def _emit_json(stream: object, value: object) -> None:
    _write_bytes(stream, dumps_canonical_json(value))


def _write_bytes(stream: object, value: bytes) -> None:
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write(value)
        return
    write = getattr(stream, "write")
    write(value.decode("utf-8"))
