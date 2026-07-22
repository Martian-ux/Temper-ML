from __future__ import annotations

import json
from pathlib import Path
from threading import Event
import time

import pytest

import temper_ml.app_services.consumer_journey as consumer_journey_module
from temper_ml.app_services.consumer_journey import (
    ConsumerJourneyError,
    ConsumerJourneyService,
    HuggingFaceDatasetClient,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.library_backend import LibraryCapability
from temper_ml.runtime.library_double import DeterministicLibraryBackend
from temper_ml.store.evidence import TypedEvidenceStore


class WordTokenizer:
    def __init__(self, source: Path, identity: ContentIdentity) -> None:
        del source
        self._identity = identity

    @property
    def identity(self) -> ContentIdentity:
        return self._identity

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class InterruptOnceBackend(DeterministicLibraryBackend):
    def __init__(self, capability: LibraryCapability) -> None:
        super().__init__(capability)
        self._interrupt_once = True

    def train(self, **values):
        if not self._interrupt_once:
            return super().train(**values)
        interrupted = False
        original_progress = values["on_progress"]
        original_interruption = values["interruption_requested"]

        def on_progress(step, loss):
            nonlocal interrupted
            original_progress(step, loss)
            interrupted = True

        def interruption_requested():
            return interrupted or original_interruption()

        values["on_progress"] = on_progress
        values["interruption_requested"] = interruption_requested
        try:
            return super().train(**values)
        finally:
            self._interrupt_once = False


class BlockingBackend(DeterministicLibraryBackend):
    def __init__(self, capability: LibraryCapability) -> None:
        super().__init__(capability)
        self.started = Event()
        self.release = Event()

    def train(self, **values):
        self.started.set()
        assert self.release.wait(timeout=10)
        return super().train(**values)


def _capability() -> LibraryCapability:
    return LibraryCapability(
        accelerator_backend="cpu",
        accelerator_architecture="synthetic-library-cpu",
        accelerator_model="Synthetic library CPU",
        accelerator_count=0,
        accelerator_memory_bytes=(),
        system_memory_bytes=16_000_000_000,
        supported_precision_modes=("fp32",),
        supported_quantization_modes=("none",),
        capabilities=(
            "accelerate",
            "cancellation",
            "checkpoint_resume",
            "evaluation_inference",
            "local_staging",
            "local_use_inference",
            "lora",
            "peft",
            "transformers",
        ),
        library_versions={
            "accelerate": "test",
            "peft": "test",
            "torch": "test",
            "transformers": "test",
        },
    )


def _journey(
    tmp_path: Path,
    backend: DeterministicLibraryBackend | None = None,
) -> tuple[ConsumerJourneyService, DeterministicLibraryBackend]:
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.mkdir()
    (model / "weights.test").write_bytes(b"synthetic-model")
    (tokenizer / "tokenizer.test").write_bytes(b"synthetic-tokenizer")
    backend = backend or DeterministicLibraryBackend(_capability())
    journey = ConsumerJourneyService(
        tmp_path / "project",
        backend_factory=lambda setup: backend,
        tokenizer_factory=WordTokenizer,
    )
    journey.setup_project(
        mode="real_local",
        model_source=str(model.resolve()),
        tokenizer_source=str(tokenizer.resolve()),
    )
    return journey, backend


def _rows() -> bytes:
    return json.dumps(
        [
            {
                "context": f"Context {index}",
                "completion": f"Completion {index}",
                "cot": f"Reasoning {index}",
                "output": f"Output {index}",
            }
            for index in range(20)
        ]
    ).encode()


def _wait_for_terminal(journey: ConsumerJourneyService) -> dict[str, object]:
    deadline = time.monotonic() + 30
    workspace = journey.workspace()
    while workspace["operation"]["status"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
        workspace = journey.workspace()
    return workspace


def test_real_consumer_journey_trains_and_labels_library_artifact(
    tmp_path: Path,
) -> None:
    journey, backend = _journey(tmp_path)
    imported = journey.import_dataset(
        source_format="json",
        source_bytes=_rows(),
        options={"maximum_tokens": 128, "train_weight": 20, "validation_weight": 1},
    )
    assert imported["analysis"]["accepted_rows"] == 20
    resolved = journey.resolve_candidates(
        {"training_steps": 2, "sequence_length": 32, "target_modules": "c_attn"}
    )
    assert resolved["demo"] is False
    assert resolved["preflight"]["status"] == "ready"
    assert journey.real.sources is not None
    assert journey.real.sources.staging_root.is_relative_to(
        journey.project_root / ".temper" / "derived"
    )
    assert journey._session_path.is_relative_to(
        journey.project_root / ".temper" / "derived"
    )

    assert journey.launch_candidates()["started"] is True
    workspace = _wait_for_terminal(journey)
    assert workspace["operation"]["status"] == "completed"
    assert workspace["artifacts"][0]["artifact_kind"] == "real_trained_lora_adapter"
    assert workspace["artifacts"][0]["demo"] is False
    assert backend.train_calls == 1

    inference = journey.focused_local_use(
        candidate_key="selected", prompt="Local real-path prompt", save=False
    )
    assert inference["demo"] is False
    assert inference["artifact_label"] == "Verified real trained LoRA adapter"
    assert backend.inference_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dropout", "not-a-number"),
        ("dropout", "1"),
        ("learning_rate", "NaN"),
        ("learning_rate", "0"),
    ),
)
def test_real_recipe_decimal_options_fail_actionably(
    tmp_path: Path, field: str, value: str
) -> None:
    journey, _ = _journey(tmp_path)
    journey.import_dataset(
        source_format="json",
        source_bytes=_rows(),
        options={"maximum_tokens": 128, "train_weight": 20, "validation_weight": 1},
    )

    with pytest.raises(ConsumerJourneyError) as captured:
        journey.resolve_candidates({field: value})

    assert captured.value.code == "request_option_invalid"
    assert captured.value.details == {"field": field}


def test_fixture_workspace_is_unmistakably_demo_labeled(tmp_path: Path) -> None:
    journey = ConsumerJourneyService(tmp_path)
    journey.setup_project(mode="fixture_demo")
    journey.import_dataset()
    journey.resolve_candidates()

    resolved_workspace = journey.workspace()
    assert len(resolved_workspace["resolutions"]) == 2
    assert (
        next(
            stage for stage in resolved_workspace["stages"] if stage["key"] == "recipe"
        )["complete"]
        is True
    )

    journey.launch_candidates()

    workspace = journey.workspace()
    assert workspace["demo"] is True
    assert "no model is trained" in workspace["mode_label"].lower()
    assert all(artifact["demo"] is True for artifact in workspace["artifacts"])
    assert all(
        "not a trained adapter" in artifact["label"]
        for artifact in workspace["artifacts"]
    )


def test_fixture_local_file_bytes_remain_a_demo_import(tmp_path: Path) -> None:
    journey = ConsumerJourneyService(tmp_path)
    journey.setup_project(mode="fixture_demo")
    source = json.dumps(
        [
            {
                "instruction": "Local fixture instruction",
                "context": "Local fixture context",
                "response": "Local fixture response",
            }
        ]
    ).encode()

    imported = journey.import_dataset(source_format="json", source_bytes=source)

    assert imported["statistics"]["accepted_rows"] == 1
    assert journey.workspace()["demo"] is True


def test_fixture_local_results_repeat_demo_and_not_trained_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journey = ConsumerJourneyService(tmp_path)
    journey.setup_project(mode="fixture_demo")
    monkeypatch.setattr(
        journey.demo,
        "focused_local_use",
        lambda **_values: {"status": "completed"},
    )
    monkeypatch.setattr(
        journey.demo,
        "batch_local_use",
        lambda **_values: {"status": "completed"},
    )
    monkeypatch.setattr(
        journey.demo,
        "export_selected",
        lambda **_values: {"status": "verified"},
    )

    focused = journey.focused_local_use(candidate_key="ember", prompt="demo")
    batch = journey.batch_local_use(candidate_key="ember", prompts=("demo",))
    exported = journey.export_selected(candidate_key="ember")

    assert focused["demo"] is True
    assert batch["demo"] is True
    assert exported["demo"] is True
    assert "not a trained adapter" in focused["artifact_label"]
    assert "no model inference" in batch["inference_label"]
    assert "not a trained adapter" in exported["export_label"]


def test_local_file_import_applies_explicit_bounded_row_prefix(tmp_path: Path) -> None:
    journey, _ = _journey(tmp_path)

    imported = journey.import_dataset(
        source_format="json",
        source_bytes=_rows(),
        options={
            "row_limit": 5,
            "maximum_tokens": 128,
            "train_weight": 5,
            "validation_weight": 1,
        },
    )

    assert imported["analysis"]["source_rows"] == 5
    assert imported["source"]["available_rows"] == 20
    assert imported["source"]["imported_rows"] == 5
    assert imported["source"]["row_limit"] == 5


def test_real_attempts_sessions_and_exports_are_retry_safe(tmp_path: Path) -> None:
    journey, backend = _journey(tmp_path)
    first = journey.import_dataset(
        source_format="json",
        source_bytes=_rows(),
        options={"maximum_tokens": 128, "train_weight": 20, "validation_weight": 1},
    )
    second = journey.import_dataset(
        source_format="json",
        source_bytes=_rows(),
        options={"maximum_tokens": 128, "train_weight": 20, "validation_weight": 1},
    )
    assert (
        first["dataset_version"]["logical_id"]
        != second["dataset_version"]["logical_id"]
    )

    journey.resolve_candidates({"training_steps": 1, "sequence_length": 32})
    journey.resolve_candidates({"training_steps": 1, "sequence_length": 32})
    journey.launch_candidates()
    assert _wait_for_terminal(journey)["operation"]["status"] == "completed"

    with pytest.raises(ConsumerJourneyError) as consumed:
        journey.launch_candidates()
    assert consumed.value.code == "real_run_attempt_consumed"

    journey.resolve_candidates({"training_steps": 1, "sequence_length": 32})
    journey.launch_candidates()
    assert _wait_for_terminal(journey)["operation"]["status"] == "completed"
    assert backend.train_calls == 2

    journey.focused_local_use(
        candidate_key="selected", prompt="Saved real prompt", save=True
    )
    journey.focused_local_use(
        candidate_key="selected", prompt="Saved real prompt", save=True
    )
    journey.batch_local_use(
        candidate_key="selected",
        prompts=("First real batch prompt", "Second real batch prompt"),
        save=True,
    )
    first_export = journey.export_selected(candidate_key="selected")
    second_export = journey.export_selected(candidate_key="selected")

    workspace = journey.workspace()
    assert workspace["local_use"] == {"saved_session_count": 3, "export_count": 2}
    assert len(workspace["saved_sessions"]) == 3
    assert len(workspace["verified_exports"]) == 2
    assert (
        first_export["adapter_export"]["logical_id"]
        != second_export["adapter_export"]["logical_id"]
    )


def test_workspace_polling_does_not_catalog_store_during_active_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = BlockingBackend(_capability())
    journey, _ = _journey(tmp_path, backend)
    journey.import_dataset(
        source_format="json",
        source_bytes=_rows(),
        options={"maximum_tokens": 128, "train_weight": 20, "validation_weight": 1},
    )
    journey.resolve_candidates({"training_steps": 1, "sequence_length": 32})

    original = TypedEvidenceStore.iter_records

    def guarded_iter_records(store: TypedEvidenceStore):
        if backend.started.is_set() and not backend.release.is_set():
            raise AssertionError("active writer must use the pre-launch store snapshot")
        return original(store)

    monkeypatch.setattr(TypedEvidenceStore, "iter_records", guarded_iter_records)
    try:
        journey.launch_candidates()
        assert backend.started.wait(timeout=10)
        active = journey.workspace()
        assert active["operation"]["status"] == "running"
        assert active["store"]["status"] == "write_in_progress"
        assert active["store"]["snapshot_during_active_write"] is True
    finally:
        backend.release.set()
    assert _wait_for_terminal(journey)["operation"]["status"] == "completed"


def test_real_consumer_cancellation_reaches_a_truthful_terminal_state(
    tmp_path: Path,
) -> None:
    backend = BlockingBackend(_capability())
    journey, _ = _journey(tmp_path, backend)
    journey.import_dataset(
        source_format="json",
        source_bytes=_rows(),
        options={"maximum_tokens": 128, "train_weight": 20, "validation_weight": 1},
    )
    journey.resolve_candidates({"training_steps": 2, "sequence_length": 32})

    journey.launch_candidates()
    assert backend.started.wait(timeout=10)
    cancelled = journey.cancel_run()
    assert cancelled["accepted"] is True
    assert cancelled["operation"]["cancellation_requested"] is True
    backend.release.set()

    workspace = _wait_for_terminal(journey)
    assert workspace["operation"]["status"] == "cancelled"
    assert workspace["artifacts"] == []
    assert "Resolve again" in workspace["operation"]["recovery_action"]


def test_real_private_session_restores_mode_without_claiming_artifact(
    tmp_path: Path,
) -> None:
    journey, backend = _journey(tmp_path)
    restored = ConsumerJourneyService(
        journey.project_root,
        backend_factory=lambda setup: backend,
        tokenizer_factory=WordTokenizer,
    )

    workspace = restored.workspace()
    assert workspace["mode"] == "real_local"
    assert workspace["recovery_required"] is True
    assert workspace["operation"]["status"] == "interrupted"
    assert workspace["dataset"] is None
    assert workspace["artifacts"] == []
    assert "No prior artifact" in workspace["operation"]["recovery_action"]


def test_hugging_face_source_mode_rejects_ambiguous_fields() -> None:
    client = HuggingFaceDatasetClient()
    url = "https://huggingface.co/datasets/Glint-Research/Fable-5-traces"

    with pytest.raises(ConsumerJourneyError) as repository_conflict:
        client.fetch(
            dataset_url=url,
            config="pi_agent",
            split="train",
            file_path="fable5_cot_merged.jsonl",
            row_limit=4,
            source_mode="repository_file",
        )
    assert repository_conflict.value.code == "hugging_face_source_fields_conflict"

    with pytest.raises(ConsumerJourneyError) as split_conflict:
        client.fetch(
            dataset_url=url,
            config="pi_agent",
            split="train",
            file_path="fable5_cot_merged.jsonl",
            row_limit=4,
            source_mode="config_split",
        )
    assert split_conflict.value.code == "hugging_face_source_fields_conflict"


def test_hugging_face_viewer_rows_are_bounded_and_invalid_json_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = {
        "num_rows_total": 3,
        "rows": [
            {"row": {"context": f"context-{index}", "completion": "done"}}
            for index in range(3)
        ],
    }
    monkeypatch.setattr(
        consumer_journey_module,
        "_read_public_url",
        lambda _url, _limit: json.dumps(rows).encode(),
    )
    client = HuggingFaceDatasetClient()

    imported, provenance = client.fetch(
        dataset_url="https://huggingface.co/datasets/example/public-data",
        config="default",
        split="train",
        file_path=None,
        row_limit=1,
    )

    assert len(imported.rows) == 1
    assert provenance["imported_rows"] == 1
    assert provenance["available_rows"] == 3

    monkeypatch.setattr(
        consumer_journey_module,
        "_read_public_url",
        lambda _url, _limit: b"not-json",
    )
    with pytest.raises(ConsumerJourneyError) as captured:
        client.fetch(
            dataset_url="https://huggingface.co/datasets/example/public-data",
            config="default",
            split="train",
            file_path=None,
            row_limit=1,
        )
    assert captured.value.code == "hugging_face_rows_invalid"
    assert captured.value.details["action"]


def test_real_dataset_preview_is_bounded_but_rendered_bytes_are_complete(
    tmp_path: Path,
) -> None:
    journey, _ = _journey(tmp_path)
    rows = json.dumps(
        [
            {"context": "x" * 5_000, "completion": f"completion-{index}"}
            for index in range(20)
        ]
    ).encode()

    imported = journey.import_dataset(
        source_format="json",
        source_bytes=rows,
        options={
            "maximum_tokens": 128,
            "train_weight": 20,
            "validation_weight": 1,
        },
    )

    preview = imported["previews"][0]
    assert preview["text_truncated"] is True
    assert len(preview["text"]) < preview["full_text_characters"]
    assert imported["rendered_bytes_count"] > len(preview["text"].encode())


def test_interrupted_real_run_resumes_from_retained_checkpoint(
    tmp_path: Path,
) -> None:
    backend = InterruptOnceBackend(_capability())
    journey, _ = _journey(tmp_path, backend)
    journey.import_dataset(
        source_format="json",
        source_bytes=_rows(),
        options={"maximum_tokens": 128, "train_weight": 20, "validation_weight": 1},
    )
    journey.resolve_candidates({"training_steps": 2, "sequence_length": 32})

    journey.launch_candidates()
    interrupted = _wait_for_terminal(journey)
    assert interrupted["operation"]["status"] == "interrupted"
    assert "Launch again" in interrupted["operation"]["recovery_action"]

    resumed = journey.launch_candidates()
    assert resumed["recovery"] is True
    assert _wait_for_terminal(journey)["operation"]["status"] == "completed"
    assert backend.train_calls == 2
