"""Deterministic focused and batch inference for verified fixture adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.records import (
    FrozenJsonObject,
    RecordValidationError,
    freeze_json_object,
    identity_fields,
    require_non_negative_int,
    require_positive_int,
    thaw_json,
)
from temper_ml.store.canonical_json import dumps_canonical_json

FIXTURE_INFERENCE_PROJECTION = HashProjection("runtime.fixture_inference", "v1")
FIXTURE_INFERENCE_INPUT_PROJECTION = HashProjection(
    "runtime.fixture_inference_input", "v1"
)
FIXTURE_INFERENCE_OUTPUT_PROJECTION = HashProjection(
    "runtime.fixture_inference_output", "v1"
)
FIXTURE_INFERENCE_EVIDENCE_PROJECTION = HashProjection(
    "runtime.fixture_inference_evidence", "v1"
)
FIXTURE_INFERENCE_IDENTITY = content_identity(
    FIXTURE_INFERENCE_PROJECTION,
    {
        "schema_version": "v1",
        "runtime": "temper_fixture_inference",
        "deterministic": True,
        "network": False,
        "accelerator": False,
    },
)


class FixtureInferenceError(RuntimeError):
    """A stable, public-safe inference failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class InferenceSettings:
    """Exact supported settings for the deterministic fixture runtime."""

    temperature: int = 0
    maximum_tokens: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or self.temperature != 0:
            raise FixtureInferenceError("inference_temperature_not_deterministic")
        try:
            require_positive_int("maximum_tokens", self.maximum_tokens)
            require_non_negative_int("seed", self.seed)
        except RecordValidationError:
            raise FixtureInferenceError("inference_settings_invalid") from None

    def to_dict(self) -> dict[str, int]:
        return {
            "temperature": self.temperature,
            "maximum_tokens": self.maximum_tokens,
            "seed": self.seed,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "InferenceSettings":
        if not isinstance(value, Mapping) or set(value) != {
            "temperature",
            "maximum_tokens",
            "seed",
        }:
            raise FixtureInferenceError("inference_settings_invalid")
        try:
            return cls(
                temperature=value["temperature"],  # type: ignore[arg-type]
                maximum_tokens=value["maximum_tokens"],  # type: ignore[arg-type]
                seed=value["seed"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            raise FixtureInferenceError("inference_settings_invalid") from None


@dataclass(frozen=True)
class FixtureInferenceRequest:
    adapter_bytes: bytes
    artifact_content_identity: ContentIdentity
    settings: InferenceSettings
    inputs: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_bytes, bytes) or not self.adapter_bytes:
            raise FixtureInferenceError("inference_adapter_bytes_invalid")
        if not isinstance(self.artifact_content_identity, ContentIdentity):
            raise FixtureInferenceError("inference_artifact_identity_invalid")
        if not isinstance(self.settings, InferenceSettings):
            raise FixtureInferenceError("inference_settings_invalid")
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise FixtureInferenceError("inference_inputs_invalid")
        frozen: list[FrozenJsonObject] = []
        try:
            for index, value in enumerate(self.inputs):
                frozen.append(
                    freeze_json_object(value, field=f"inference_inputs[{index}]")
                )
        except (RecordValidationError, TypeError, ValueError):
            raise FixtureInferenceError("inference_inputs_invalid") from None
        object.__setattr__(self, "inputs", tuple(frozen))


@dataclass(frozen=True)
class FixtureInferenceResult:
    inputs: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    settings: InferenceSettings
    runtime_identity: ContentIdentity
    runtime_evidence: ContentIdentity

    def __post_init__(self) -> None:
        if len(self.inputs) != len(self.outputs) or not self.inputs:
            raise FixtureInferenceError("inference_result_invalid")

    def to_view(self) -> dict[str, object]:
        return {
            "settings": self.settings.to_dict(),
            "outputs": [thaw_json(value) for value in self.outputs],
            "runtime_identity": identity_fields(self.runtime_identity),
            "runtime_evidence": identity_fields(self.runtime_evidence),
        }


class FixtureInferenceRuntime:
    """Offline deterministic inference over already verified adapter bytes."""

    runtime_identity = FIXTURE_INFERENCE_IDENTITY

    def infer(self, request: FixtureInferenceRequest) -> FixtureInferenceResult:
        if not isinstance(request, FixtureInferenceRequest):
            raise FixtureInferenceError("inference_request_invalid")
        adapter_identity = ContentIdentity(
            "sha256", hashlib.sha256(request.adapter_bytes).hexdigest()
        )
        outputs: list[FrozenJsonObject] = []
        input_identities: list[ContentIdentity] = []
        output_identities: list[ContentIdentity] = []
        for frozen_input in request.inputs:
            value = thaw_json(frozen_input)
            input_identity = content_identity(FIXTURE_INFERENCE_INPUT_PROJECTION, value)
            digest = hashlib.sha256(
                dumps_canonical_json(
                    {
                        "schema_version": "v1",
                        "adapter_identity": identity_fields(adapter_identity),
                        "artifact_content_identity": identity_fields(
                            request.artifact_content_identity
                        ),
                        "input_identity": identity_fields(input_identity),
                        "settings": request.settings.to_dict(),
                    }
                )
            ).hexdigest()
            output = freeze_json_object(
                {
                    "text": f"fixture-response-{digest[:24]}",
                    "finish_reason": "fixture_complete",
                    "input_identity": identity_fields(input_identity),
                },
                field="inference_output",
            )
            outputs.append(output)
            input_identities.append(input_identity)
            output_identities.append(
                content_identity(
                    FIXTURE_INFERENCE_OUTPUT_PROJECTION,
                    thaw_json(output),
                )
            )
        evidence = content_identity(
            FIXTURE_INFERENCE_EVIDENCE_PROJECTION,
            {
                "schema_version": "v1",
                "runtime_identity": identity_fields(FIXTURE_INFERENCE_IDENTITY),
                "adapter_identity": identity_fields(adapter_identity),
                "artifact_content_identity": identity_fields(
                    request.artifact_content_identity
                ),
                "settings": request.settings.to_dict(),
                "input_identities": [
                    identity_fields(identity) for identity in input_identities
                ],
                "output_identities": [
                    identity_fields(identity) for identity in output_identities
                ],
            },
        )
        return FixtureInferenceResult(
            request.inputs,
            tuple(outputs),
            request.settings,
            FIXTURE_INFERENCE_IDENTITY,
            evidence,
        )
