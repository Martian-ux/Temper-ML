import json
from pathlib import Path
import shutil
import subprocess

from temper_ml.cli import main
from temper_ml.domain.artifacts import file_identity
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore
from temper_ml.store.redaction import RedactionContext


REPO_ROOT = Path(__file__).parents[2]
FIXTURE = REPO_ROOT / "fixtures" / "projects" / "minimal"


def _store(root: Path) -> TypedEvidenceStore:
    return TypedEvidenceStore(root, redaction_context=RedactionContext())


def test_committed_minimal_project_exercises_all_slice_one_commands(
    capsys,
) -> None:
    records = _store(FIXTURE).iter_records()
    task = next(
        stored for stored in records if stored.envelope.record_type == "task_definition"
    )
    assert task.record.rendering_contract == file_identity(
        FIXTURE / "sources" / "rendering-contract.txt"
    )

    for command in ("status", "verify", "dump"):
        assert main([command, str(FIXTURE)]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == dumps_canonical_json(json.loads(captured.out)).decode()

    assert (
        main(
            [
                "manifest",
                str(FIXTURE),
                "--type",
                "project",
                "--id",
                "project-minimal",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert '"record_type":"project"' in captured.out


def test_fixture_reconstructs_after_interrupted_and_corrupt_derived_writes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    store = _store(project)
    reconstructed = store.reconstruct()
    canonical = {
        stored.path: stored.path.read_bytes() for stored in reconstructed.records
    }
    for stream in reconstructed.streams:
        canonical.update(
            {event.path: event.path.read_bytes() for event in stream.events}
        )

    record = reconstructed.records[0]
    interrupted = record.path.with_name(f".{record.path.name}.{'c' * 32}.tmp")
    interrupted.write_bytes(b"partial interrupted write")
    assert store.verify().record_count == 2

    expected = store.rebuild_stream_state("project-lifecycle")
    state = store.layout.stream_state("project-lifecycle")
    state.write_bytes(b"corrupt rebuildable bytes")
    assert store.rebuild_stream_state("project-lifecycle") == expected
    state.unlink()
    assert store.rebuild_stream_state("project-lifecycle") == expected

    assert {path: path.read_bytes() for path in canonical} == canonical
    assert store.verify().event_count == 1
    assert store.public_dump().value["classification"] == "public_projection"


def test_fixture_gitignore_exception_is_an_exact_file_allowlist() -> None:
    if shutil.which("git") is None:
        return
    ignored = (
        "fixtures/projects/minimal/.temper/private-secret.json",
        "fixtures/projects/minimal/.temper/derived/streams/x/state.json",
        (
            "fixtures/projects/minimal/.temper/immutable/records/project/"
            "sha256/unexpected.json"
        ),
    )
    for path in ignored:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0

    committed = next(
        (FIXTURE / ".temper" / "immutable" / "records" / "project" / "sha256").iterdir()
    )
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "--no-index",
            committed.relative_to(REPO_ROOT).as_posix(),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 1
    attributes = subprocess.run(
        [
            "git",
            "check-attr",
            "eol",
            "--",
            committed.relative_to(REPO_ROOT).as_posix(),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert attributes.stdout.rstrip().endswith("eol: lf")
