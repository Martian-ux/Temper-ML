from pathlib import Path

from temper_ml.domain.projections import (
    HashProjection,
    content_identity,
    projection_preimage,
)
from temper_ml.store.canonical_json import loads_canonical_json


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
