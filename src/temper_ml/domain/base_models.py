"""Exact base-model revision contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordValidationError,
    TypedRecord,
    identity_fields,
    require_identifier,
    require_text,
)


@dataclass(frozen=True)
class BaseModelRevision(TypedRecord):
    """One exact model and tokenizer revision, independent of friendly names."""

    RECORD_TYPE: ClassVar[str] = "base_model_revision"

    model_id: str
    display_name: str
    model_family: str
    architecture: str
    source: str
    revision: str
    weights_identity: ContentIdentity
    tokenizer_identity: ContentIdentity
    license: str

    def __post_init__(self) -> None:
        require_identifier("model_id", self.model_id)
        require_identifier("source", self.source)
        for field in (
            "display_name",
            "model_family",
            "architecture",
            "revision",
            "license",
        ):
            require_text(field, getattr(self, field))
        for field in ("weights_identity", "tokenizer_identity"):
            if not isinstance(getattr(self, field), ContentIdentity):
                raise RecordValidationError(f"{field} must be a content identity")

    def to_payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "model_family": self.model_family,
            "architecture": self.architecture,
            "source": self.source,
            "revision": self.revision,
            "weights_identity": identity_fields(self.weights_identity),
            "tokenizer_identity": identity_fields(self.tokenizer_identity),
            "license": self.license,
        }
