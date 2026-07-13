import hashlib

import pytest

from temper_ml.domain.projections import ContentIdentity, content_identity
from temper_ml.domain.records import identity_fields, thaw_json
from temper_ml.runtime.fixture_inference import (
    FIXTURE_INFERENCE_EVIDENCE_PROJECTION,
    FIXTURE_INFERENCE_IDENTITY,
    FIXTURE_INFERENCE_INPUT_PROJECTION,
    FIXTURE_INFERENCE_OUTPUT_PROJECTION,
    FixtureInferenceError,
    FixtureInferenceRequest,
    FixtureInferenceRuntime,
    InferenceSettings,
)


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def test_fixture_inference_is_repeatable_and_binds_exact_settings_and_inputs() -> None:
    inputs = ({"text": "Synthetic alpha"}, {"text": "Synthetic beta"})
    request = FixtureInferenceRequest(
        adapter_bytes=b"TEMPER-FIXTURE-ADAPTER-v1\n" + b"a" * 32 + b"\n",
        artifact_content_identity=_identity("artifact"),
        settings=InferenceSettings(temperature=0, maximum_tokens=32, seed=7),
        inputs=inputs,
    )
    runtime = FixtureInferenceRuntime()

    first = runtime.infer(request)
    second = runtime.infer(request)

    assert first == second
    assert first.runtime_identity == FIXTURE_INFERENCE_IDENTITY
    assert len(first.outputs) == 2
    assert first.outputs[0] != first.outputs[1]
    assert first.to_view()["settings"] == {
        "temperature": 0,
        "maximum_tokens": 32,
        "seed": 7,
    }


def test_fixture_inference_copies_inputs_and_rejects_nondeterministic_settings() -> (
    None
):
    mutable = {"text": "Synthetic prompt"}
    request = FixtureInferenceRequest(
        adapter_bytes=b"adapter",
        artifact_content_identity=_identity("artifact"),
        settings=InferenceSettings(),
        inputs=(mutable,),
    )
    before = FixtureInferenceRuntime().infer(request)
    mutable["text"] = "Changed after request construction"

    assert FixtureInferenceRuntime().infer(request) == before
    with pytest.raises(
        FixtureInferenceError, match="inference_temperature_not_deterministic"
    ):
        InferenceSettings(temperature=1)
    with pytest.raises(FixtureInferenceError, match="inference_settings_invalid"):
        InferenceSettings.from_mapping({"temperature": 0})


def test_per_input_output_is_stable_across_focused_batch_and_reordered_batch() -> None:
    adapter_bytes = b"TEMPER-FIXTURE-ADAPTER-v1\n" + b"b" * 32 + b"\n"
    artifact_identity = _identity("stable-artifact")
    settings = InferenceSettings(temperature=0, maximum_tokens=48, seed=11)
    alpha = {"text": "Synthetic alpha"}
    beta = {"text": "Synthetic beta"}
    runtime = FixtureInferenceRuntime()

    focused = runtime.infer(
        FixtureInferenceRequest(
            adapter_bytes,
            artifact_identity,
            settings,
            (alpha,),
        )
    )
    batch = runtime.infer(
        FixtureInferenceRequest(
            adapter_bytes,
            artifact_identity,
            settings,
            (alpha, beta),
        )
    )
    reordered = runtime.infer(
        FixtureInferenceRequest(
            adapter_bytes,
            artifact_identity,
            settings,
            (beta, alpha),
        )
    )

    assert focused.outputs[0] == batch.outputs[0] == reordered.outputs[1]
    assert batch.outputs[1] == reordered.outputs[0]
    assert batch.runtime_evidence != reordered.runtime_evidence
    assert FIXTURE_INFERENCE_INPUT_PROJECTION != FIXTURE_INFERENCE_OUTPUT_PROJECTION

    adapter_identity = ContentIdentity(
        "sha256", hashlib.sha256(adapter_bytes).hexdigest()
    )
    input_identities = tuple(
        content_identity(FIXTURE_INFERENCE_INPUT_PROJECTION, value)
        for value in (alpha, beta)
    )
    output_identities = tuple(
        content_identity(FIXTURE_INFERENCE_OUTPUT_PROJECTION, thaw_json(value))
        for value in batch.outputs
    )
    expected_evidence = content_identity(
        FIXTURE_INFERENCE_EVIDENCE_PROJECTION,
        {
            "schema_version": "v1",
            "runtime_identity": identity_fields(FIXTURE_INFERENCE_IDENTITY),
            "adapter_identity": identity_fields(adapter_identity),
            "artifact_content_identity": identity_fields(artifact_identity),
            "settings": settings.to_dict(),
            "input_identities": [
                identity_fields(identity) for identity in input_identities
            ],
            "output_identities": [
                identity_fields(identity) for identity in output_identities
            ],
        },
    )
    assert batch.runtime_evidence == expected_evidence
