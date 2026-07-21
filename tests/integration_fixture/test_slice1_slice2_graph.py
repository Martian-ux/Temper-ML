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
from temper_ml.domain.datasets import (
    RENDERED_BYTES_FORMAT,
    AcceptedExample,
    DatasetAdapter,
    DatasetStatistics,
    DatasetVersion,
    DeduplicationRule,
    FieldMapping,
    FilterRule,
    PreviewSelection,
    RendererSpec,
    SourceDescriptor,
    SplitCount,
    SplitMembership,
    SplitPart,
    SplitRule,
    rendered_example_identity,
    renderer_identity,
    split_membership_identity,
)
from temper_ml.domain.experiments import Experiment, derive_experiment
from temper_ml.domain.evaluations import (
    ArtifactIntegrityStatus,
    CaseSuiteKind,
    ComparisonOperator,
    ConfidenceLabel,
    ConfidenceRule,
    EvaluationCase,
    EvaluationResult,
    EvaluationSuite,
    EvaluatorKind,
    EvaluatorSpec,
    EvidenceStatus,
    HardQualifier,
    MetricDirection,
    MetricObservation,
    OptimizationObjective,
    RecommendationPolicy,
    Review,
    ReviewCandidate,
    ReviewEntry,
    ReviewMode,
    ReviewOutput,
    ReviewRating,
    ReviewStage,
    SuiteEvidenceState,
    UserDecision,
    UserDecisionStatus,
    build_recommendation,
)
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
    identity_fields,
    record_reference,
)
from temper_ml.domain.retention import (
    CleanupObjectReceipt,
    CleanupObjectStatus,
    CleanupOutcome,
    CleanupReceipt,
)
from temper_ml.domain.runs import EvaluationMode, ResolvedRuntimeRequest, Run
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore
from temper_ml.store.redaction import RedactionContext


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _complete_record_graph() -> tuple[TypedRecord, ...]:
    tokenizer = _identity("graph-tokenizer")
    source_bytes = dumps_canonical_json(
        [
            {
                "instruction": "Synthetic graph input",
                "response": "Synthetic graph output",
            }
        ]
    )
    field_mapping = FieldMapping("instruction", "response")
    renderer = RendererSpec()
    rendered_text = (
        "### Instruction\nSynthetic graph input\n### Response\nSynthetic graph output"
    )
    rendered_identity = rendered_example_identity(rendered_text)
    split_rule = SplitRule(1, (SplitPart("train", 1),))
    split_membership = (SplitMembership(rendered_identity, "train"),)
    rendered_bytes = dumps_canonical_json(
        {
            "rendered_identity": identity_fields(rendered_identity),
            "source_ordinal": 1,
            "split": "train",
            "text": rendered_text,
        }
    )
    token_count = len(rendered_text.split())
    dataset = DatasetVersion(
        version_id="dataset-graph",
        source=SourceDescriptor(
            DatasetAdapter.JSON,
            ContentIdentity("sha256", hashlib.sha256(source_bytes).hexdigest()),
            1,
        ),
        field_mapping=field_mapping,
        renderer=renderer,
        renderer_identity=renderer_identity(field_mapping, renderer),
        filter_rule=FilterRule(),
        deduplication_rule=DeduplicationRule(),
        tokenizer_identity=tokenizer,
        split_rule=split_rule,
        split_identity=split_membership_identity(split_rule, split_membership),
        rendered_bytes_format=RENDERED_BYTES_FORMAT,
        rendered_bytes_identity=ContentIdentity(
            "sha256", hashlib.sha256(rendered_bytes).hexdigest()
        ),
        rendered_bytes_count=len(rendered_bytes),
        preview_limit=1,
        preview_selections=(
            PreviewSelection(1, rendered_identity, "train", token_count),
        ),
        accepted_examples=(
            AcceptedExample(
                1,
                rendered_identity,
                len(rendered_text.encode("utf-8")),
                token_count,
            ),
        ),
        split_membership=split_membership,
        exclusions=(),
        statistics=DatasetStatistics(
            1,
            1,
            0,
            0,
            token_count,
            token_count,
            token_count,
            (SplitCount("train", 1),),
        ),
    )
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
    runtime_identity = _identity("runtime")
    training_state_identity = _identity("training-state")
    request = ResolvedRuntimeRequest(
        request_id="request-graph",
        experiment=record_reference(parent),
        experiment_manifest_identity=parent.manifest_identity,
        recipe_resolution=record_reference(resolution),
        dataset_version_identity=dataset.identity,
        rendered_dataset_identity=dataset.rendered_bytes_identity,
        rendered_dataset_byte_count=dataset.rendered_bytes_count,
        hardware_capability_profile=record_reference(profile),
        execution_target=record_reference(target),
        runtime_identity=runtime_identity,
        preflight_identity=_identity("preflight"),
        training_state_identity=training_state_identity,
        evaluation_mode=EvaluationMode.NO_QUALITY_EVALUATION,
        training_steps=resolution.training_steps,
        starting_step=0,
    )
    run = Run(
        run_id="run-graph",
        experiment=record_reference(parent),
        experiment_manifest_identity=parent.manifest_identity,
        attempt_number=1,
        hardware_capability_profile=record_reference(profile),
        execution_target=record_reference(target),
        runtime_identity=runtime_identity,
        request_identity=request.identity,
        training_state_identity=training_state_identity,
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
    evaluation_suite = EvaluationSuite(
        "suite-graph",
        CaseSuiteKind.CONFIRMATION,
        SuiteEvidenceState.UNSEALED,
        (EvaluationCase("case-graph", _identity("evaluation-case-graph")),),
        (
            EvaluatorSpec(
                "accuracy-check",
                EvaluatorKind.TASK_METRIC,
                "accuracy",
                MetricDirection.MAXIMIZE,
            ),
        ),
    )
    review = Review(
        "review-graph",
        ReviewMode.SOLO,
        ReviewStage.RECORDED,
        (
            ReviewEntry(
                "review-prompt-graph",
                {"text": "Synthetic evaluation prompt"},
                {"temperature": 0, "maximum_tokens": 32},
                (
                    ReviewOutput(
                        "candidate-graph",
                        {"text": "Synthetic evaluation output"},
                    ),
                ),
                "The synthetic output satisfies the fixture criterion.",
                (ReviewRating("candidate-graph", "task_fit", 1),),
            ),
        ),
        "I reviewed the synthetic fixture evidence.",
        (ReviewCandidate("candidate-graph", record_reference(artifact)),),
        False,
    )
    evaluation_result = EvaluationResult(
        "result-graph",
        record_reference(artifact),
        EvaluationMode.FULL_SUITE,
        ArtifactIntegrityStatus.PASSED,
        artifact.integrity_evidence,
        EvidenceStatus.PASSED,
        metrics=(
            MetricObservation(
                "accuracy",
                EvaluatorKind.TASK_METRIC,
                1,
                MetricDirection.MAXIMIZE,
            ),
        ),
        suite=record_reference(evaluation_suite),
        suite_state=evaluation_suite.state,
        review=record_reference(review),
    )
    recommendation_policy = RecommendationPolicy(
        "recommendation-policy-graph",
        (
            HardQualifier(
                "accuracy",
                ComparisonOperator.GREATER_THAN_OR_EQUAL,
                1,
            ),
        ),
        (),
        (OptimizationObjective("accuracy", MetricDirection.MAXIMIZE),),
        (),
        (
            ConfidenceRule(
                ConfidenceLabel.HIGH,
                (EvidenceStatus.PASSED,),
                (SuiteEvidenceState.UNSEALED,),
                1,
            ),
        ),
    )
    recommendation = build_recommendation(
        "recommendation-graph",
        recommendation_policy,
        (evaluation_result,),
    )
    user_decision = UserDecision(
        "decision-graph",
        record_reference(recommendation),
        record_reference(artifact),
        UserDecisionStatus.SELECTED,
        evaluation_result.evidence_status,
    )
    cleanup_object = CleanupObjectReceipt(
        entry_id="entry-cleanup-graph",
        logical_key="logs/cleanup-graph.log",
        byte_class="debugging_evidence",
        byte_count=1,
        content_identity=_identity("cleanup-object-graph"),
        status=CleanupObjectStatus.REMOVED,
        physical_bytes_freed=True,
        subjects=(),
    )
    cleanup_receipt = CleanupReceipt(
        receipt_id="cleanup-receipt-00000000000000000000000000000000",
        execution_id="cleanup-execution-00000000000000000000000000000000",
        project=record_reference(project),
        inventory_identity=_identity("cleanup-inventory-graph"),
        plan_identity=_identity("cleanup-plan-graph"),
        outcome=CleanupOutcome.COMPLETED,
        selected_entry_ids=(cleanup_object.entry_id,),
        objects=(cleanup_object,),
        logical_bytes_removed=1,
        physical_bytes_freed=1,
        impact_categories=("debugging_evidence",),
        affected_subjects=(),
        availability_updates=(),
    )
    return (
        dataset,
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
        request,
        run,
        artifact,
        availability,
        session,
        exported,
        evaluation_suite,
        evaluation_result,
        recommendation_policy,
        recommendation,
        user_decision,
        review,
        cleanup_receipt,
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
