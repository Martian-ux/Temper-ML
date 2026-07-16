from pathlib import Path

from temper_ml.domain.projections import (
    HashProjection,
    content_identity,
    projection_preimage,
)
from temper_ml.domain.evaluations import (
    CASE_MEMBERSHIP_PROJECTION,
    CaseSuiteKind,
    EvaluationCase,
    EvaluationSuite,
    EvaluatorKind,
    EvaluatorSpec,
    MetricDirection,
    SuiteEvidenceState,
)
from temper_ml.domain.records import CORE_PROJECTION_REGISTRY
from temper_ml.store.canonical_json import loads_canonical_json
from temper_ml.runtime.fixture_adapter import FIXTURE_RUNTIME_IDENTITY
from temper_ml.runtime.fixture_inference import FIXTURE_INFERENCE_IDENTITY


FIXTURE = Path(__file__).parents[1] / "fixtures" / "identity" / "project-policy-v1.json"


def test_content_identity_uses_explicit_projection_version_and_domain_prefix():
    projected_fields = loads_canonical_json(FIXTURE.read_bytes())
    projection = HashProjection(name="project_policy", version="v1")

    preimage = projection_preimage(projection, projected_fields)
    identity = content_identity(projection, projected_fields)

    assert preimage.startswith(b"temper:project_policy@v1\n")
    assert identity.algorithm == "sha256"
    assert (
        identity.value
        == "db67380147829e194febebc4d1a67c8ee12f19fda03cacc7c9bc3d18493c472f"
    )
    assert str(identity) == (
        "sha256:db67380147829e194febebc4d1a67c8ee12f19fda03cacc7c9bc3d18493c472f"
    )


def test_fixture_runtime_identities_are_versioned_and_stable():
    assert FIXTURE_RUNTIME_IDENTITY.algorithm == "sha256"
    assert FIXTURE_INFERENCE_IDENTITY.algorithm == "sha256"
    assert FIXTURE_RUNTIME_IDENTITY != FIXTURE_INFERENCE_IDENTITY


def test_slice_six_projection_registrations_and_case_membership_are_versioned():
    record_types = {
        "evaluation_result",
        "evaluation_suite",
        "recommendation",
        "recommendation_policy",
        "review",
        "user_decision",
    }
    registrations = {
        registration.record_type: registration
        for registration in CORE_PROJECTION_REGISTRY.registrations
        if registration.record_type in record_types
    }
    assert set(registrations) == record_types
    assert {
        registration.projection.label for registration in registrations.values()
    } == {f"record.{record_type}@v1" for record_type in record_types}

    case_identity = content_identity(HashProjection("fixture.case", "v1"), {"v": 1})
    suite = EvaluationSuite(
        "suite-projection",
        CaseSuiteKind.DEVELOPMENT,
        SuiteEvidenceState.UNSEALED,
        (EvaluationCase("case-projection", case_identity),),
        (
            EvaluatorSpec(
                "evaluator-projection",
                EvaluatorKind.FORMAT_CHECK,
                "format_validity",
                MetricDirection.MAXIMIZE,
            ),
        ),
    )
    assert CASE_MEMBERSHIP_PROJECTION.label == "evaluation.case_membership@v1"
    assert suite.to_payload()["case_membership_identity"] == {
        "algorithm": "sha256",
        "value": suite.case_membership_identity.value,
    }
