import hashlib
import json
from pathlib import Path

import pytest

from temper_ml.domain.projections import (
    HashProjection,
    ProjectionError,
    ProjectionRegistration,
    ProjectionRegistry,
    content_identity,
)
from temper_ml.domain.records import (
    CORE_LOGICAL_ID_FIELDS,
    CORE_PROJECTION_REGISTRY,
    RecordEnvelope,
    RecordValidationError,
    record_reference,
)
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.domain.projections import ContentIdentity


REPO_ROOT = Path(__file__).parents[2]


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _task(schema: dict | None = None) -> TaskDefinition:
    return TaskDefinition(
        task_id="task-rewrite",
        display_name="Synthetic rewrite",
        description="Rewrite synthetic input while retaining named entities.",
        input_schema=schema or {"required": ["input"]},
        output_schema={"required": ["output"]},
        rendering_contract=_identity("renderer"),
        objectives=("entity_preservation", "style_match"),
        capabilities=("text_generation",),
    )


def test_envelope_round_trip_verifies_registered_projection_and_identity() -> None:
    envelope = _task().to_envelope()

    assert envelope.record_type == "task_definition"
    assert envelope.schema_version == "v1"
    assert envelope.projection.label == "record.task_definition@v1"
    assert RecordEnvelope.from_dict(envelope.to_dict()) == envelope

    tampered = envelope.to_dict()
    tampered["payload"]["description"] = "Tampered"
    with pytest.raises(RecordValidationError, match="identity mismatch"):
        RecordEnvelope.from_dict(tampered)


def test_record_payload_is_deeply_alias_safe() -> None:
    source = {"properties": {"input": {"type": "string"}}}
    task = _task(source)
    before = task.identity

    source["properties"]["input"]["type"] = "integer"

    assert task.identity == before
    assert task.to_payload()["input_schema"]["properties"]["input"]["type"] == "string"
    with pytest.raises(RecordValidationError, match="must match"):
        record_reference(task, "task-alias")


def test_projection_registry_rejects_duplicates_and_unknown_versions() -> None:
    registration = ProjectionRegistration(
        "synthetic_record", "v1", HashProjection("record.synthetic", "v1")
    )
    with pytest.raises(ProjectionError, match="duplicate"):
        ProjectionRegistry((registration, registration))

    with pytest.raises(ProjectionError, match="unknown"):
        CORE_PROJECTION_REGISTRY.resolve("project", "v2")


def test_envelope_rejects_counterfeit_or_rehashed_invalid_typed_payload() -> None:
    class CounterfeitRun:
        RECORD_TYPE = "run"
        SCHEMA_VERSION = "v1"

        def to_payload(self):
            return {}

    with pytest.raises(RecordValidationError, match="concrete type"):
        RecordEnvelope.from_record(CounterfeitRun())

    registration = CORE_PROJECTION_REGISTRY.resolve("run", "v1")
    projected = {
        "record_type": "run",
        "schema_version": "v1",
        "projection_version": "v1",
        "payload": {},
    }
    invalid = {
        **projected,
        "identity": {
            "algorithm": "sha256",
            "value": content_identity(registration.projection, projected).value,
        },
    }
    with pytest.raises(RecordValidationError, match="fields are invalid"):
        RecordEnvelope.from_dict(invalid)


def test_public_projection_manifest_matches_code_and_record_schemas() -> None:
    manifest_path = (
        REPO_ROOT / "schemas" / "identity-projections" / "core-domain-v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registrations = manifest["registrations"]
    code = CORE_PROJECTION_REGISTRY.registrations

    assert [item["record_type"] for item in registrations] == [
        item.record_type for item in code
    ]
    assert set(CORE_LOGICAL_ID_FIELDS) == {item.record_type for item in code}
    for serialized, registered in zip(registrations, code):
        assert serialized["record_schema_version"] == registered.schema_version
        assert serialized["projection_name"] == registered.projection.name
        assert serialized["projection_version"] == registered.projection.version
        schema_path = REPO_ROOT / serialized["record_schema"]
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["allOf"][1]["properties"]["record_type"]["const"] == (
            registered.record_type
        )
        logical_id_field = CORE_LOGICAL_ID_FIELDS[registered.record_type]
        assert (
            logical_id_field in schema["allOf"][1]["properties"]["payload"]["required"]
        )


def test_published_schemas_pin_reference_types_and_zero_accelerator_parity() -> None:
    records = REPO_ROOT / "schemas" / "records"
    common = json.loads((records / "common.schema.json").read_text(encoding="utf-8"))
    task_ref = common["$defs"]["taskDefinitionReference"]
    assert task_ref["allOf"][1]["properties"]["record_type"]["const"] == (
        "task_definition"
    )
    dataset_ref = common["$defs"]["datasetVersionReference"]
    assert dataset_ref["allOf"][1]["properties"]["record_type"]["const"] == (
        "dataset_version"
    )
    runtime_request_ref = common["$defs"]["resolvedRuntimeRequestReference"]
    assert (
        runtime_request_ref["allOf"][1]["properties"]["record_type"]["const"]
        == "resolved_runtime_request"
    )
    for name, record_type in (
        ("evaluationResultReference", "evaluation_result"),
        ("evaluationSuiteReference", "evaluation_suite"),
        ("recommendationReference", "recommendation"),
        ("recommendationPolicyReference", "recommendation_policy"),
        ("reviewReference", "review"),
        ("userDecisionReference", "user_decision"),
    ):
        assert (
            common["$defs"][name]["allOf"][1]["properties"]["record_type"]["const"]
            == record_type
        )

    experiment = json.loads(
        (records / "experiment.schema.json").read_text(encoding="utf-8")
    )
    properties = experiment["allOf"][1]["properties"]["payload"]["properties"]
    assert properties["task_definition"]["$ref"].endswith("/taskDefinitionReference")
    assert properties["recipe"]["$ref"].endswith("/recipeReference")

    hardware = json.loads(
        (records / "hardware_capability_profile.schema.json").read_text(
            encoding="utf-8"
        )
    )
    capability = hardware["allOf"][1]["properties"]["payload"]["properties"]
    assert capability["accelerator_count"]["minimum"] == 0
    assert "minItems" not in capability["accelerator_memory_bytes"]
