from decimal import Decimal
import hashlib
from pathlib import Path
import re

from temper_ml.domain.artifacts import (
    Artifact,
    ArtifactAvailability,
    ArtifactContentKind,
    AvailabilityState,
    StorageReference,
)
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import CompatibilityGroup, RuntimeTargetConstraint
from temper_ml.domain.experiments import Experiment, derive_experiment
from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.local_use import AdapterExport, LocalUseSession
from temper_ml.domain.policies import BaselinePolicy, PerModelBaseline
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import (
    CORE_PROJECTION_REGISTRY,
    TypedRecord,
    record_reference,
)
from temper_ml.domain.runs import Run
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore
from temper_ml.store.redaction import RedactionContext


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _complete_record_graph() -> tuple[TypedRecord, ...]:
    tokenizer = _identity("graph-tokenizer")
    task = TaskDefinition(
        "task-graph",
        "Synthetic graph task",
        "Exercise every Slice 2 record through Slice 1 evidence services.",
        {"required": ["input"]},
        {"required": ["output"]},
        _identity("graph-renderer"),
        ("determinism",),
        ("text_generation",),
    )
    model = BaseModelRevision(
        "model-graph",
        "Synthetic model",
        "synthetic-family",
        "synthetic-causal-lm",
        "public_fixture",
        "revision-a",
        _identity("graph-weights"),
        tokenizer,
        "Apache-2.0",
    )
    project = Project(
        "project-graph",
        "Synthetic graph project",
        "Exercise the complete public contract graph.",
        record_reference(task),
        (record_reference(model),),
    )
    baseline = BaselinePolicy(
        "baseline-graph", (PerModelBaseline(_identity("comparison-policy")),)
    )
    policy = ProjectPolicy(
        "policy-graph",
        record_reference(project),
        record_reference(task),
        task.rendering_contract,
        _identity("evaluation-policy"),
        (_identity("case-suite"),),
        _identity("readiness-policy"),
        _identity("retention-policy"),
        ("balanced",),
        record_reference(baseline),
        _identity("recommendation-policy"),
    )
    target = ExecutionTarget(
        "target-graph",
        "wsl2_rocm",
        "linux",
        "rocm",
        _identity("runtime-contract"),
        ("bf16", "lora"),
        {"minimum_runtime_version": "v1"},
    )
    requirements = HardwareRequirements(
        "requirements-graph",
        ("wsl2_rocm",),
        ("rocm",),
        8_000_000_000,
        16_000_000_000,
        ("bf16",),
        (),
        ("lora",),
        {"maximum_sequence_length": 2048},
    )
    profile = HardwareCapabilityProfile(
        "profile-graph",
        record_reference(target),
        "fixture",
        "none",
        "No accelerator",
        0,
        (),
        32_000_000_000,
        ("bf16",),
        (),
        ("lora",),
        {"fixture_runtime": "v1"},
    )
    group = CompatibilityGroup(
        "group-graph",
        record_reference(model),
        tokenizer,
        task.rendering_contract,
        "lora",
        ("q_proj", "v_proj"),
        (
            RuntimeTargetConstraint(
                target.target_class,
                target.accelerator_backend,
                target.runtime_contract,
                target.capabilities,
            ),
        ),
        ("linear",),
    )
    recipe = Recipe(
        "recipe-graph",
        "balanced",
        "v1",
        "balanced",
        "small",
        "standard",
        "none",
        "short",
        "periodic",
        "light",
        "full",
        {},
    )
    resolution = RecipeResolution(
        "resolution-graph",
        record_reference(recipe),
        record_reference(model),
        record_reference(requirements),
        record_reference(target),
        "lora",
        ("q_proj", "v_proj"),
        8,
        16,
        Decimal("0.05"),
        Decimal("0.0002"),
        8,
        1024,
        "adamw",
        "bf16",
        4,
        7,
        "cosine",
        100,
        25,
        "none",
        {"transformers": "v1", "peft": "v1"},
        ("memory_budget",),
    )
    common = {
        "project": record_reference(project),
        "project_policy": record_reference(policy),
        "task_definition": record_reference(task),
        "base_model_revision": record_reference(model),
        "tokenizer_identity": tokenizer,
        "recipe": record_reference(recipe),
        "recipe_resolution": record_reference(resolution),
        "evaluation_policy": policy.evaluation_policy,
        "compatibility_group": record_reference(group),
        "hardware_requirements": record_reference(requirements),
        "execution_target": record_reference(target),
    }
    parent = Experiment(
        experiment_id="experiment-parent",
        dataset_version=_identity("dataset-a"),
        **common,
    )
    derived = Experiment(
        experiment_id="experiment-derived",
        dataset_version=_identity("dataset-b"),
        **common,
    )
    derivation = derive_experiment(
        parent,
        derived,
        derivation_id="derivation-graph",
        diff_id="diff-graph",
        reason_code="dataset_revision",
        reason="Use the second committed synthetic dataset revision.",
    )
    run = Run(
        "run-graph",
        record_reference(parent),
        parent.manifest_identity,
        1,
        record_reference(profile),
        record_reference(target),
        _identity("runtime"),
        _identity("request"),
        _identity("training-state"),
    )
    storage = StorageReference("project_store", "adapter_primary")
    artifact = Artifact(
        "artifact-graph",
        record_reference(project),
        record_reference(run),
        "lora",
        ArtifactContentKind.BUNDLE,
        _identity("adapter-content"),
        record_reference(model),
        tokenizer,
        (record_reference(group),),
        (),
        (storage,),
        _identity("integrity"),
        _identity("provenance"),
        _identity("lineage"),
    )
    availability = ArtifactAvailability(
        "availability-graph",
        record_reference(artifact),
        AvailabilityState.AVAILABLE,
        ("final_adapter",),
        (storage,),
        False,
        artifact.content_identity,
    )
    session = LocalUseSession(
        "session-graph",
        record_reference(project),
        record_reference(artifact),
        artifact.content_identity,
        record_reference(model),
        tokenizer,
        record_reference(group),
        record_reference(target),
        {"temperature": 0, "maximum_tokens": 64},
        ({"text": "Committed synthetic prompt"},),
        ({"text": "Committed synthetic response"},),
        _identity("runtime-evidence"),
        artifact.integrity_evidence,
    )
    exported = AdapterExport(
        "export-graph",
        record_reference(artifact),
        artifact.content_identity,
        _identity("export-manifest"),
        artifact.integrity_evidence,
        (record_reference(group),),
        {"adapter_type": "lora", "target_class": "wsl2_rocm"},
        artifact.provenance,
        "temper_adapter_bundle",
        StorageReference("export_store", "adapter_export"),
    )
    return (
        task,
        model,
        project,
        baseline,
        policy,
        target,
        requirements,
        profile,
        group,
        recipe,
        resolution,
        parent,
        derived,
        derivation.manifest_diff,
        derivation,
        run,
        artifact,
        availability,
        session,
        exported,
    )


def test_every_slice_two_record_round_trips_through_slice_one(tmp_path: Path) -> None:
    store = TypedEvidenceStore(tmp_path, redaction_context=RedactionContext())
    records = _complete_record_graph()
    expected_types = {
        registration.record_type
        for registration in CORE_PROJECTION_REGISTRY.registrations
    }
    assert {record.RECORD_TYPE for record in records} == expected_types

    for record in reversed(records):
        store.write_record(record)

    reconstructed = store.reconstruct()
    assert len(reconstructed.records) == len(records)
    assert {
        stored.envelope.identity: stored.record for stored in reconstructed.records
    } == {record.identity: record for record in records}
    assert set(store.verify().record_counts) == expected_types

    public_bytes = dumps_canonical_json(store.public_dump().value)
    assert re.search(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", public_bytes) is None
    assert b"Committed synthetic prompt" not in public_bytes
    assert b"adapter_primary" not in public_bytes
    assert b"revision-a" not in public_bytes
    for private_canary in (
        b"determinism",
        b"text_generation",
        b"synthetic-family",
        b"q_proj",
        b"memory_budget",
        b"lora",
    ):
        assert private_canary not in public_bytes
