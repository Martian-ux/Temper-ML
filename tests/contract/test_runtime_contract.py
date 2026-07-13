import hashlib

import pytest

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    record_reference,
)
from temper_ml.domain.runs import EvaluationMode, ResolvedRuntimeRequest, Run


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(kind: str, logical_id: str) -> RecordReference:
    return RecordReference(kind, logical_id, _identity(f"{kind}:{logical_id}"))


def _request(**changes) -> ResolvedRuntimeRequest:
    values = {
        "request_id": "request-fixture-one",
        "experiment": _reference("experiment", "experiment-fixture"),
        "experiment_manifest_identity": _identity("experiment-manifest"),
        "recipe_resolution": _reference("recipe_resolution", "resolution-fixture"),
        "dataset_version_identity": _identity("dataset-version"),
        "rendered_dataset_identity": _identity("rendered-dataset"),
        "rendered_dataset_byte_count": 123,
        "hardware_capability_profile": _reference(
            "hardware_capability_profile", "profile-fixture"
        ),
        "execution_target": _reference("execution_target", "target-fixture"),
        "runtime_identity": _identity("runtime"),
        "preflight_identity": _identity("preflight"),
        "training_state_identity": _identity("training-state-zero"),
        "evaluation_mode": EvaluationMode.NO_QUALITY_EVALUATION,
        "training_steps": 4,
        "starting_step": 0,
    }
    values.update(changes)
    return ResolvedRuntimeRequest(**values)


def test_resolved_runtime_request_is_immutable_registered_and_complete() -> None:
    request = _request()
    decoded = request.to_envelope().to_record()

    assert decoded == request
    assert request.to_payload()["evaluation_mode"] == "no_quality_evaluation"
    assert request.to_payload()["resume_from_run"] is None
    assert request.to_payload()["resume_checkpoint_identity"] is None
    assert request.identity == _request().identity


def test_resume_request_requires_a_bound_run_checkpoint_and_nonzero_step() -> None:
    prior = _reference("run", "run-interrupted")
    checkpoint = _identity("checkpoint")
    resumed = _request(
        request_id="request-fixture-recovery",
        starting_step=2,
        training_state_identity=_identity("training-state-two"),
        resume_from_run=prior,
        resume_checkpoint_identity=checkpoint,
    )

    assert resumed.resume_from_run == prior
    assert resumed.resume_checkpoint_identity == checkpoint
    with pytest.raises(RecordValidationError, match="supplied together"):
        _request(resume_from_run=prior)
    with pytest.raises(RecordValidationError, match="required exactly"):
        _request(
            resume_from_run=prior,
            resume_checkpoint_identity=checkpoint,
        )


def test_run_attempts_pin_request_state_and_later_attempts_require_lineage() -> None:
    request = _request()
    common = {
        "experiment": request.experiment,
        "experiment_manifest_identity": request.experiment_manifest_identity,
        "hardware_capability_profile": request.hardware_capability_profile,
        "execution_target": request.execution_target,
        "runtime_identity": request.runtime_identity,
        "request_identity": request.identity,
        "training_state_identity": request.training_state_identity,
    }
    first = Run(run_id="run-fixture-one", attempt_number=1, **common)
    retry = Run(
        run_id="run-fixture-two",
        attempt_number=2,
        retry_of=record_reference(first),
        **common,
    )

    assert first.identity != retry.identity
    assert "status" not in first.to_payload()
    with pytest.raises(RecordValidationError, match="later attempts"):
        Run(run_id="run-invalid", attempt_number=2, **common)
