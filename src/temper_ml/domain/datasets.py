"""Immutable contracts for deterministic LLM-training dataset versions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.records import (
    RecordValidationError,
    TypedRecord,
    identity_fields,
    require_identifier,
    require_non_negative_int,
    require_positive_int,
)

RENDERER_IDENTITY_PROJECTION = HashProjection("dataset.renderer", "v1")
RENDERED_EXAMPLE_PROJECTION = HashProjection("dataset.rendered_example", "v1")
SPLIT_IDENTITY_PROJECTION = HashProjection("dataset.split_membership", "v1")
RENDERED_BYTES_FORMAT = "temper.dataset.rendered-jsonl@v1"


class DatasetAdapter(str, Enum):
    """Explicit, local-only source adapter contracts."""

    JSON = "json@v1"
    JSONL = "jsonl@v1"
    CSV = "csv@v1"
    HUGGING_FACE_ROWS = "hugging_face_rows@v1"


class RendererKind(str, Enum):
    """Temper-owned deterministic renderer contracts."""

    INSTRUCTION_RESPONSE = "instruction_response"
    TRACE_COMPLETION = "trace_completion"
    TRACE_COMPONENTS = "trace_components"


class DeduplicationMode(str, Enum):
    """Supported rendered-example deduplication behavior."""

    EXACT_RENDERED_TEXT = "exact_rendered_text"


class DeduplicationKeep(str, Enum):
    """Tie-breaking behavior for duplicates."""

    FIRST_SOURCE_ORDER = "first_source_order"


class ExclusionPhase(str, Enum):
    """Stable pipeline phases recorded by safe exclusion receipts."""

    VALIDATION = "validation"
    FILTERING = "filtering"
    DEDUPLICATION = "deduplication"


@dataclass(frozen=True)
class FieldMapping:
    """Direct source-field mapping into the renderer's logical roles."""

    instruction_field: str
    response_field: str
    context_field: str | None = None
    cot_field: str | None = None
    output_field: str | None = None

    def __post_init__(self) -> None:
        require_identifier("instruction_field", self.instruction_field)
        require_identifier("response_field", self.response_field)
        if self.context_field is not None:
            require_identifier("context_field", self.context_field)
        if self.cot_field is not None:
            require_identifier("cot_field", self.cot_field)
        if self.output_field is not None:
            require_identifier("output_field", self.output_field)
        fields = tuple(
            field
            for field in (
                self.instruction_field,
                self.context_field,
                self.response_field,
                self.cot_field,
                self.output_field,
            )
            if field is not None
        )
        if len(set(fields)) != len(fields):
            raise RecordValidationError("field mapping source fields must be unique")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "instruction_field": self.instruction_field,
            "response_field": self.response_field,
            "context_field": self.context_field,
        }
        if self.cot_field is not None:
            value["cot_field"] = self.cot_field
        if self.output_field is not None:
            value["output_field"] = self.output_field
        return value


@dataclass(frozen=True)
class RendererSpec:
    """Versioned renderer selection with no implicit template mutation."""

    kind: RendererKind = RendererKind.INSTRUCTION_RESPONSE
    version: str = "v1"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RendererKind):
            raise RecordValidationError("renderer kind is invalid")
        if self.version != "v1":
            raise RecordValidationError("renderer version is unsupported")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "version": self.version}


@dataclass(frozen=True)
class SourceDescriptor:
    """Path-free provenance for the exact locally supplied source bytes or rows."""

    adapter: DatasetAdapter
    source_identity: ContentIdentity
    row_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, DatasetAdapter):
            raise RecordValidationError("dataset adapter is invalid")
        if not isinstance(self.source_identity, ContentIdentity):
            raise RecordValidationError("source_identity must be a content identity")
        require_non_negative_int("row_count", self.row_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter.value,
            "source_identity": identity_fields(self.source_identity),
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class FilterRule:
    """Deterministic bounds applied to rendered Unicode text and token counts."""

    minimum_characters: int = 1
    maximum_characters: int | None = None
    maximum_tokens: int | None = None

    def __post_init__(self) -> None:
        require_non_negative_int("minimum_characters", self.minimum_characters)
        if self.maximum_characters is not None:
            require_positive_int("maximum_characters", self.maximum_characters)
            if self.maximum_characters < self.minimum_characters:
                raise RecordValidationError(
                    "maximum_characters must not be below minimum_characters"
                )
        if self.maximum_tokens is not None:
            require_positive_int("maximum_tokens", self.maximum_tokens)

    def to_dict(self) -> dict[str, int | None]:
        return {
            "minimum_characters": self.minimum_characters,
            "maximum_characters": self.maximum_characters,
            "maximum_tokens": self.maximum_tokens,
        }


@dataclass(frozen=True)
class DeduplicationRule:
    """Exact rendered-text deduplication with an explicit tie-breaker."""

    mode: DeduplicationMode = DeduplicationMode.EXACT_RENDERED_TEXT
    keep: DeduplicationKeep = DeduplicationKeep.FIRST_SOURCE_ORDER

    def __post_init__(self) -> None:
        if self.mode is not DeduplicationMode.EXACT_RENDERED_TEXT:
            raise RecordValidationError("deduplication mode is unsupported")
        if self.keep is not DeduplicationKeep.FIRST_SOURCE_ORDER:
            raise RecordValidationError("deduplication keep rule is unsupported")

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode.value, "keep": self.keep.value}


@dataclass(frozen=True)
class SplitPart:
    """One named deterministic split and its integer allocation weight."""

    name: str
    weight: int

    def __post_init__(self) -> None:
        require_identifier("split name", self.name)
        require_positive_int("split weight", self.weight)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "weight": self.weight}


@dataclass(frozen=True)
class SplitRule:
    """Identity-bound weighted hash partitioning rule."""

    seed: int
    parts: tuple[SplitPart, ...]
    algorithm: str = "sha256_weighted_bucket@v1"

    def __post_init__(self) -> None:
        require_non_negative_int("split seed", self.seed)
        if self.algorithm != "sha256_weighted_bucket@v1":
            raise RecordValidationError("split algorithm is unsupported")
        if not isinstance(self.parts, tuple) or not self.parts:
            raise RecordValidationError("split parts must be a non-empty tuple")
        if any(not isinstance(part, SplitPart) for part in self.parts):
            raise RecordValidationError("split parts contain an invalid value")
        names = tuple(part.name for part in self.parts)
        if len(set(names)) != len(names):
            raise RecordValidationError("split names must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "parts": [part.to_dict() for part in self.parts],
        }


@dataclass(frozen=True)
class ExclusionReceipt:
    """Public-safe row disposition that never contains rejected source values."""

    source_ordinal: int
    phase: ExclusionPhase
    reason_code: str
    retained_source_ordinal: int | None = None

    def __post_init__(self) -> None:
        require_positive_int("source_ordinal", self.source_ordinal)
        if not isinstance(self.phase, ExclusionPhase):
            raise RecordValidationError("exclusion phase is invalid")
        require_identifier("reason_code", self.reason_code)
        if self.retained_source_ordinal is not None:
            require_positive_int(
                "retained_source_ordinal", self.retained_source_ordinal
            )
            if self.phase is not ExclusionPhase.DEDUPLICATION:
                raise RecordValidationError(
                    "only duplicate receipts may identify a retained row"
                )
            if self.retained_source_ordinal >= self.source_ordinal:
                raise RecordValidationError(
                    "duplicate retained row must precede the excluded row"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_ordinal": self.source_ordinal,
            "phase": self.phase.value,
            "reason_code": self.reason_code,
            "retained_source_ordinal": self.retained_source_ordinal,
        }


@dataclass(frozen=True)
class AcceptedExample:
    """Value-free evidence for one accepted rendered example."""

    source_ordinal: int
    rendered_identity: ContentIdentity
    rendered_utf8_bytes: int
    token_count: int

    def __post_init__(self) -> None:
        require_positive_int("source_ordinal", self.source_ordinal)
        if not isinstance(self.rendered_identity, ContentIdentity):
            raise RecordValidationError("rendered_identity must be a content identity")
        require_positive_int("rendered_utf8_bytes", self.rendered_utf8_bytes)
        require_non_negative_int("token_count", self.token_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_ordinal": self.source_ordinal,
            "rendered_identity": identity_fields(self.rendered_identity),
            "rendered_utf8_bytes": self.rendered_utf8_bytes,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class PreviewSelection:
    """Value-free evidence identifying one deterministically selected preview."""

    source_ordinal: int
    rendered_identity: ContentIdentity
    split: str
    token_count: int

    def __post_init__(self) -> None:
        require_positive_int("source_ordinal", self.source_ordinal)
        if not isinstance(self.rendered_identity, ContentIdentity):
            raise RecordValidationError("rendered_identity must be a content identity")
        require_identifier("split", self.split)
        require_non_negative_int("token_count", self.token_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_ordinal": self.source_ordinal,
            "rendered_identity": identity_fields(self.rendered_identity),
            "split": self.split,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class SplitMembership:
    """Exact accepted-content membership in one named split."""

    rendered_identity: ContentIdentity
    split: str

    def __post_init__(self) -> None:
        if not isinstance(self.rendered_identity, ContentIdentity):
            raise RecordValidationError("rendered_identity must be a content identity")
        require_identifier("split", self.split)

    def to_dict(self) -> dict[str, object]:
        return {
            "rendered_identity": identity_fields(self.rendered_identity),
            "split": self.split,
        }


@dataclass(frozen=True)
class SplitCount:
    """Stable summary count for one configured split."""

    split: str
    count: int

    def __post_init__(self) -> None:
        require_identifier("split", self.split)
        require_non_negative_int("split count", self.count)

    def to_dict(self) -> dict[str, object]:
        return {"split": self.split, "count": self.count}


@dataclass(frozen=True)
class DatasetStatistics:
    """Exact integer-only summary of a prepared version."""

    source_rows: int
    accepted_rows: int
    excluded_rows: int
    duplicate_rows: int
    total_tokens: int
    minimum_tokens: int
    maximum_tokens: int
    split_counts: tuple[SplitCount, ...]

    def __post_init__(self) -> None:
        for field in (
            "source_rows",
            "accepted_rows",
            "excluded_rows",
            "duplicate_rows",
            "total_tokens",
            "minimum_tokens",
            "maximum_tokens",
        ):
            require_non_negative_int(field, getattr(self, field))
        if self.source_rows != self.accepted_rows + self.excluded_rows:
            raise RecordValidationError("dataset row statistics do not balance")
        if self.duplicate_rows > self.excluded_rows:
            raise RecordValidationError("duplicate count exceeds exclusions")
        if not isinstance(self.split_counts, tuple) or not self.split_counts:
            raise RecordValidationError("split_counts must be a non-empty tuple")
        if any(not isinstance(item, SplitCount) for item in self.split_counts):
            raise RecordValidationError("split_counts contains an invalid value")
        if sum(item.count for item in self.split_counts) != self.accepted_rows:
            raise RecordValidationError("split counts do not match accepted rows")
        if self.accepted_rows == 0:
            if any((self.total_tokens, self.minimum_tokens, self.maximum_tokens)):
                raise RecordValidationError("empty token statistics must be zero")
        elif self.minimum_tokens > self.maximum_tokens:
            raise RecordValidationError("token statistic bounds are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_rows": self.source_rows,
            "accepted_rows": self.accepted_rows,
            "excluded_rows": self.excluded_rows,
            "duplicate_rows": self.duplicate_rows,
            "total_tokens": self.total_tokens,
            "minimum_tokens": self.minimum_tokens,
            "maximum_tokens": self.maximum_tokens,
            "split_counts": [item.to_dict() for item in self.split_counts],
        }


def renderer_identity(
    field_mapping: FieldMapping, renderer: RendererSpec
) -> ContentIdentity:
    """Identify every governed input to exact deterministic rendering."""

    if not isinstance(field_mapping, FieldMapping) or not isinstance(
        renderer, RendererSpec
    ):
        raise RecordValidationError("renderer identity inputs are invalid")
    return content_identity(
        RENDERER_IDENTITY_PROJECTION,
        {
            "field_mapping": field_mapping.to_dict(),
            "renderer": renderer.to_dict(),
        },
    )


def rendered_example_identity(text: str) -> ContentIdentity:
    """Identify one exact rendered training example."""

    if not isinstance(text, str) or not text:
        raise RecordValidationError("rendered text must not be empty")
    return content_identity(RENDERED_EXAMPLE_PROJECTION, {"text": text})


def split_membership_identity(
    rule: SplitRule, membership: tuple[SplitMembership, ...]
) -> ContentIdentity:
    """Identify the complete split rule and resulting exact membership."""

    if not isinstance(rule, SplitRule) or not isinstance(membership, tuple):
        raise RecordValidationError("split identity inputs are invalid")
    return content_identity(
        SPLIT_IDENTITY_PROJECTION,
        {
            "rule": rule.to_dict(),
            "membership": [item.to_dict() for item in membership],
        },
    )


@dataclass(frozen=True)
class DatasetVersion(TypedRecord):
    """One immutable, fully evidenced deterministic training dataset version."""

    RECORD_TYPE: ClassVar[str] = "dataset_version"

    version_id: str
    source: SourceDescriptor
    field_mapping: FieldMapping
    renderer: RendererSpec
    renderer_identity: ContentIdentity
    filter_rule: FilterRule
    deduplication_rule: DeduplicationRule
    tokenizer_identity: ContentIdentity
    split_rule: SplitRule
    split_identity: ContentIdentity
    rendered_bytes_format: str
    rendered_bytes_identity: ContentIdentity
    rendered_bytes_count: int
    preview_limit: int
    preview_selections: tuple[PreviewSelection, ...]
    accepted_examples: tuple[AcceptedExample, ...]
    split_membership: tuple[SplitMembership, ...]
    exclusions: tuple[ExclusionReceipt, ...]
    statistics: DatasetStatistics

    def __post_init__(self) -> None:
        require_identifier("version_id", self.version_id)
        if not isinstance(self.source, SourceDescriptor):
            raise RecordValidationError("source must be a SourceDescriptor")
        if not isinstance(self.field_mapping, FieldMapping):
            raise RecordValidationError("field_mapping must be a FieldMapping")
        if not isinstance(self.renderer, RendererSpec):
            raise RecordValidationError("renderer must be a RendererSpec")
        if self.renderer_identity != renderer_identity(
            self.field_mapping, self.renderer
        ):
            raise RecordValidationError("renderer identity mismatch")
        if not isinstance(self.filter_rule, FilterRule):
            raise RecordValidationError("filter_rule must be a FilterRule")
        if not isinstance(self.deduplication_rule, DeduplicationRule):
            raise RecordValidationError(
                "deduplication_rule must be a DeduplicationRule"
            )
        if not isinstance(self.tokenizer_identity, ContentIdentity):
            raise RecordValidationError("tokenizer_identity must be a content identity")
        if not isinstance(self.split_rule, SplitRule):
            raise RecordValidationError("split_rule must be a SplitRule")
        if self.rendered_bytes_format != RENDERED_BYTES_FORMAT:
            raise RecordValidationError("rendered bytes format is unsupported")
        if not isinstance(self.rendered_bytes_identity, ContentIdentity):
            raise RecordValidationError(
                "rendered_bytes_identity must be a content identity"
            )
        require_positive_int("rendered_bytes_count", self.rendered_bytes_count)
        require_non_negative_int("preview_limit", self.preview_limit)
        for field, expected in (
            ("preview_selections", PreviewSelection),
            ("accepted_examples", AcceptedExample),
            ("split_membership", SplitMembership),
            ("exclusions", ExclusionReceipt),
        ):
            value = getattr(self, field)
            if not isinstance(value, tuple) or any(
                not isinstance(item, expected) for item in value
            ):
                raise RecordValidationError(f"{field} contains an invalid value")
        if not self.accepted_examples:
            raise RecordValidationError("dataset version must accept at least one row")
        if tuple(
            sorted(self.accepted_examples, key=lambda item: item.source_ordinal)
        ) != (self.accepted_examples):
            raise RecordValidationError("accepted examples must be in source order")
        if tuple(sorted(self.exclusions, key=lambda item: item.source_ordinal)) != (
            self.exclusions
        ):
            raise RecordValidationError("exclusions must be in source order")
        expected_membership_order = tuple(
            sorted(
                self.split_membership,
                key=lambda item: (item.rendered_identity.value, item.split),
            )
        )
        if expected_membership_order != self.split_membership:
            raise RecordValidationError("split membership order is not canonical")
        accepted_ids = tuple(item.rendered_identity for item in self.accepted_examples)
        membership_ids = tuple(item.rendered_identity for item in self.split_membership)
        if len(set(accepted_ids)) != len(accepted_ids):
            raise RecordValidationError("accepted rendered identities must be unique")
        if len(set(membership_ids)) != len(membership_ids) or set(accepted_ids) != set(
            membership_ids
        ):
            raise RecordValidationError(
                "split membership must cover each accepted example exactly once"
            )
        split_names = {part.name for part in self.split_rule.parts}
        if any(item.split not in split_names for item in self.split_membership):
            raise RecordValidationError("split membership names are not configured")
        if self.split_identity != split_membership_identity(
            self.split_rule, self.split_membership
        ):
            raise RecordValidationError("split identity mismatch")
        membership_by_identity = {
            item.rendered_identity: item.split for item in self.split_membership
        }
        expected_preview_selections = tuple(
            PreviewSelection(
                item.source_ordinal,
                item.rendered_identity,
                membership_by_identity[item.rendered_identity],
                item.token_count,
            )
            for item in self.accepted_examples[
                : min(self.preview_limit, len(self.accepted_examples))
            ]
        )
        if self.preview_selections != expected_preview_selections:
            raise RecordValidationError(
                "preview selections must exactly match the deterministic "
                "accepted prefix"
            )
        dispositions = tuple(
            item.source_ordinal for item in self.accepted_examples
        ) + tuple(item.source_ordinal for item in self.exclusions)
        if tuple(sorted(dispositions)) != tuple(range(1, self.source.row_count + 1)):
            raise RecordValidationError(
                "source row dispositions must be complete and unique"
            )
        accepted_ordinals = {item.source_ordinal for item in self.accepted_examples}
        if any(
            receipt.retained_source_ordinal is not None
            and receipt.retained_source_ordinal not in accepted_ordinals
            for receipt in self.exclusions
        ):
            raise RecordValidationError(
                "duplicate receipt retained row is not accepted"
            )
        if not isinstance(self.statistics, DatasetStatistics):
            raise RecordValidationError("statistics must be DatasetStatistics")
        split_counts = tuple(
            SplitCount(
                part.name,
                sum(item.split == part.name for item in self.split_membership),
            )
            for part in self.split_rule.parts
        )
        tokens = tuple(item.token_count for item in self.accepted_examples)
        expected_statistics = DatasetStatistics(
            source_rows=self.source.row_count,
            accepted_rows=len(self.accepted_examples),
            excluded_rows=len(self.exclusions),
            duplicate_rows=sum(
                receipt.phase is ExclusionPhase.DEDUPLICATION
                for receipt in self.exclusions
            ),
            total_tokens=sum(tokens),
            minimum_tokens=min(tokens),
            maximum_tokens=max(tokens),
            split_counts=split_counts,
        )
        if self.statistics != expected_statistics:
            raise RecordValidationError("dataset statistics mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "version_id": self.version_id,
            "source": self.source.to_dict(),
            "field_mapping": self.field_mapping.to_dict(),
            "renderer": self.renderer.to_dict(),
            "renderer_identity": identity_fields(self.renderer_identity),
            "filter_rule": self.filter_rule.to_dict(),
            "deduplication_rule": self.deduplication_rule.to_dict(),
            "tokenizer_identity": identity_fields(self.tokenizer_identity),
            "split_rule": self.split_rule.to_dict(),
            "split_identity": identity_fields(self.split_identity),
            "rendered_bytes_format": self.rendered_bytes_format,
            "rendered_bytes_identity": identity_fields(self.rendered_bytes_identity),
            "rendered_bytes_count": self.rendered_bytes_count,
            "preview_limit": self.preview_limit,
            "preview_selections": [item.to_dict() for item in self.preview_selections],
            "accepted_examples": [item.to_dict() for item in self.accepted_examples],
            "split_membership": [item.to_dict() for item in self.split_membership],
            "exclusions": [item.to_dict() for item in self.exclusions],
            "statistics": self.statistics.to_dict(),
        }
