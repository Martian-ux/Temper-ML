import hashlib
import json
from pathlib import Path

import pytest

import temper_ml.cli as cli_module
from temper_ml.cli import main
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.projects import Project
from temper_ml.domain.records import record_reference
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest
from temper_ml.store.redaction import RedactionContext


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _fixture(root: Path) -> tuple[TaskDefinition, Project]:
    task = TaskDefinition(
        task_id="task-cli",
        display_name="Synthetic CLI task",
        description="Exercise deterministic command output.",
        input_schema={"required": ["input"]},
        output_schema={"required": ["output"]},
        rendering_contract=_identity("renderer"),
        objectives=("determinism",),
        capabilities=("text_generation",),
    )
    project = Project(
        project_id="project-cli",
        display_name="Synthetic CLI project",
        purpose="Exercise verified local commands.",
        task_definition=record_reference(task),
    )
    store = TypedEvidenceStore(root, redaction_context=RedactionContext())
    store.write_record(project)
    store.write_record(task)
    store.append_event(
        "project-lifecycle",
        EventRequest("event-cli-1", "project_created", {"synthetic": True}),
    )
    return task, project


def _assert_canonical_stdout(output: str) -> dict[str, object]:
    assert output.endswith("\n")
    assert not output.endswith("\n\n")
    value = json.loads(output)
    assert output == dumps_canonical_json(value).decode()
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize("command", ["status", "verify"])
def test_status_and_verify_emit_canonical_health_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    _fixture(tmp_path)

    assert main([command, str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    value = _assert_canonical_stdout(captured.out)
    assert value == {
        "bundle_manifest_count": 0,
        "command": command,
        "derived_state_rebuildable": True,
        "event_count": 1,
        "event_stream_count": 1,
        "record_count": 2,
        "record_counts": {"project": 1, "task_definition": 1},
        "schema_version": "v1",
        "status": "verified",
    }


def test_dump_is_public_safe_and_manifest_is_exact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task, _ = _fixture(tmp_path)

    assert main(["dump", str(tmp_path)]) == 0
    dumped = capsys.readouterr()
    value = _assert_canonical_stdout(dumped.out)
    assert dumped.err == ""
    assert value["classification"] == "public_projection"
    assert task.task_id not in dumped.out
    assert task.identity.value not in dumped.out
    assert _identity("renderer").value not in dumped.out
    assert str(tmp_path) not in dumped.out

    assert (
        main(
            [
                "manifest",
                str(tmp_path),
                "--type",
                "task_definition",
                "--id",
                task.task_id,
                "--identity",
                f"sha256:{task.identity.value}",
            ]
        )
        == 0
    )
    manifest = capsys.readouterr()
    assert manifest.err == ""
    assert _assert_canonical_stdout(manifest.out) == task.to_dict()


def test_cli_errors_are_stable_json_without_paths_or_selectors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "private-looking-project"

    assert main(["verify", str(missing)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "code": "project_not_found",
        "status": "error",
    }
    assert str(missing) not in captured.err

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["verify", str(empty)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == "store_not_found"

    _fixture(tmp_path)
    supplied = "f" * 63
    assert (
        main(
            [
                "manifest",
                str(tmp_path),
                "--type",
                "task_definition",
                "--id",
                "task-cli",
                "--identity",
                supplied,
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "code": "invalid_identity",
        "status": "error",
    }
    assert supplied not in captured.err


def test_cli_buffers_output_before_reporting_corruption(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task, _ = _fixture(tmp_path)
    path = TypedEvidenceStore(tmp_path).layout.record_path(
        task.RECORD_TYPE, task.identity
    )
    path.write_bytes(path.read_bytes() + b"\n")

    assert main(["dump", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "error"
    assert str(tmp_path) not in captured.err
    assert task.identity.value not in captured.err


def test_usage_errors_do_not_echo_supplied_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    supplied = "private-selector"

    with pytest.raises(SystemExit) as error:
        main(["manifest", supplied])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "code": "usage_error",
        "status": "error",
    }
    assert supplied not in captured.err


def test_unexpected_cli_failures_do_not_echo_exception_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    sensitive = "C:\\synthetic\\private\\unexpected"

    def fail(arguments):
        del arguments
        raise RuntimeError(sensitive)

    monkeypatch.setattr(cli_module, "_run", fail)

    assert main(["verify", str(tmp_path)]) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "code": "internal_error",
        "status": "error",
    }
    assert sensitive not in captured.err


def test_stdout_encoding_failure_uses_stable_error_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    _fixture(tmp_path)

    class AsciiOnlyOutput:
        def write(self, value):
            raise UnicodeEncodeError("ascii", value, 0, 1, "fixture encoding")

    monkeypatch.setattr(cli_module.sys, "stdout", AsciiOnlyOutput())

    assert main(["verify", str(tmp_path)]) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "code": "filesystem_error",
        "status": "error",
    }


def test_fixture_workflow_cli_fails_closed_on_unimplemented_quality_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "fixture-workflow",
                str(tmp_path),
                "--evaluation-mode",
                "light_evaluation",
            ]
        )

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "code": "usage_error",
        "status": "error",
    }
    assert str(tmp_path) not in captured.err
