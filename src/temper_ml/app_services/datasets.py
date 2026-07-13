"""Deterministic, local-only dataset import and comparison services."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from temper_ml.app_services._records import (
    require_no_conflicting_logical_revision,
    write_record_idempotently,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.domain.datasets import (
    RENDERED_BYTES_FORMAT,
    AcceptedExample,
    DatasetAdapter,
    DatasetStatistics,
    DatasetVersion,
    DeduplicationRule,
    ExclusionPhase,
    ExclusionReceipt,
    FieldMapping,
    FilterRule,
    PreviewSelection,
    RendererKind,
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
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordValidationError,
    freeze_json_value,
    identity_fields,
    require_identifier,
    require_non_negative_int,
    require_positive_int,
    thaw_json,
)
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore

_SPLIT_BUCKET_PREFIX = b"temper:dataset.split_bucket@v1\n"
_GOVERNED_CONFIGURATION_FIELDS = (
    "source.adapter",
    "field_mapping",
    "renderer",
    "filter_rule",
    "deduplication_rule",
    "tokenizer_identity",
    "split_rule",
    "preview_limit",
)
_SUMMARY_STATISTICS_FIELDS = (
    "source_rows",
    "accepted_rows",
    "excluded_rows",
    "duplicate_rows",
    "total_tokens",
    "minimum_tokens",
    "maximum_tokens",
    "split_counts",
)


class DatasetAdapterError(ValueError):
    """Stable local-adapter failure that never echoes source data."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DeterministicTokenizer(Protocol):
    """Caller-supplied tokenizer contract with an exact governed identity."""

    @property
    def identity(self) -> ContentIdentity: ...

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class ImportedSource:
    """Immutable rows and path-free source provenance from one explicit adapter."""

    descriptor: SourceDescriptor
    rows: tuple[Mapping[str, object], ...]
    _identity_preimage: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, SourceDescriptor):
            raise DatasetAdapterError("adapter_result_invalid")
        if (
            not isinstance(self.rows, tuple)
            or len(self.rows) != self.descriptor.row_count
        ):
            raise DatasetAdapterError("adapter_result_invalid")
        if not isinstance(self._identity_preimage, bytes):
            raise DatasetAdapterError("adapter_result_invalid")
        object.__setattr__(self, "_identity_preimage", bytes(self._identity_preimage))
        frozen_rows = _freeze_rows(
            self.rows,
            failure_code="adapter_result_invalid",
        )
        object.__setattr__(self, "rows", frozen_rows)


class LocalDatasetAdapter(Protocol):
    """No-network adapter interface over already local caller-supplied data."""

    adapter: DatasetAdapter

    def load(self, source: object) -> ImportedSource: ...


class JsonDatasetAdapter:
    """Load a UTF-8 JSON array of row objects from local bytes."""

    adapter = DatasetAdapter.JSON

    def load(self, source: object) -> ImportedSource:
        data = _require_source_bytes(source)
        value = _parse_json(data, failure_code="json_invalid")
        if not isinstance(value, list):
            raise DatasetAdapterError("json_rows_required")
        rows = _freeze_rows(value, failure_code="json_rows_invalid")
        return _imported_bytes_source(self.adapter, data, rows)


class JsonlDatasetAdapter:
    """Load one UTF-8 JSON row object per non-empty local byte line."""

    adapter = DatasetAdapter.JSONL

    def load(self, source: object) -> ImportedSource:
        data = _require_source_bytes(source)
        try:
            text = data.decode("utf-8")
        except UnicodeError:
            raise DatasetAdapterError("source_not_utf8") from None
        parsed: list[object] = []
        lines = text.split("\n")
        last_index = len(lines) - 1
        for index, raw_line in enumerate(lines):
            if index == last_index and not raw_line:
                continue
            line = (
                raw_line[:-1]
                if index < last_index and raw_line.endswith("\r")
                else raw_line
            )
            if not line.strip():
                raise DatasetAdapterError("jsonl_blank_line")
            parsed.append(
                _parse_json(line.encode("utf-8"), failure_code="jsonl_invalid")
            )
        rows = _freeze_rows(parsed, failure_code="jsonl_rows_invalid")
        return _imported_bytes_source(self.adapter, data, rows)


class CsvDatasetAdapter:
    """Load a UTF-8 CSV table with unique non-empty headers from local bytes."""

    adapter = DatasetAdapter.CSV

    def load(self, source: object) -> ImportedSource:
        data = _require_source_bytes(source)
        try:
            text = data.decode("utf-8")
        except UnicodeError:
            raise DatasetAdapterError("source_not_utf8") from None
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
            headers = reader.fieldnames
            if (
                headers is None
                or not headers
                or any(not header for header in headers)
                or len(set(headers)) != len(headers)
            ):
                raise DatasetAdapterError("csv_header_invalid")
            parsed_rows: list[Mapping[str, object]] = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise DatasetAdapterError("csv_row_width_invalid")
                parsed_rows.append(dict(row))
        except DatasetAdapterError:
            raise
        except (csv.Error, UnicodeError):
            raise DatasetAdapterError("csv_invalid") from None
        rows = _freeze_rows(parsed_rows, failure_code="csv_rows_invalid")
        return _imported_bytes_source(self.adapter, data, rows)


class HuggingFaceRowsDatasetAdapter:
    """Load a bounded, already-local Hugging Face-style sequence of row mappings."""

    adapter = DatasetAdapter.HUGGING_FACE_ROWS

    def load(self, source: object) -> ImportedSource:
        if isinstance(source, (str, bytes, bytearray)) or not isinstance(
            source, Sequence
        ):
            raise DatasetAdapterError("hugging_face_rows_invalid")
        rows = _freeze_rows(source, failure_code="hugging_face_rows_invalid")
        try:
            identity_bytes = _portable_json_bytes(
                [_thaw_source_value(row) for row in rows]
            )
        except (TypeError, ValueError, UnicodeError):
            raise DatasetAdapterError("hugging_face_rows_invalid") from None
        descriptor = SourceDescriptor(
            self.adapter,
            _bytes_identity(identity_bytes),
            len(rows),
        )
        return ImportedSource(descriptor, rows, identity_bytes)


@dataclass(frozen=True)
class DatasetImportRequest:
    """Complete governed inputs for one immutable dataset version."""

    version_id: str
    field_mapping: FieldMapping
    renderer: RendererSpec
    filter_rule: FilterRule
    deduplication_rule: DeduplicationRule
    split_rule: SplitRule
    tokenizer: DeterministicTokenizer
    preview_limit: int = 3

    def __post_init__(self) -> None:
        require_identifier("version_id", self.version_id)
        if not isinstance(self.field_mapping, FieldMapping):
            raise RecordValidationError("field_mapping must be a FieldMapping")
        if not isinstance(self.renderer, RendererSpec):
            raise RecordValidationError("renderer must be a RendererSpec")
        if not isinstance(self.filter_rule, FilterRule):
            raise RecordValidationError("filter_rule must be a FilterRule")
        if not isinstance(self.deduplication_rule, DeduplicationRule):
            raise RecordValidationError(
                "deduplication_rule must be a DeduplicationRule"
            )
        if not isinstance(self.split_rule, SplitRule):
            raise RecordValidationError("split_rule must be a SplitRule")
        require_non_negative_int("preview_limit", self.preview_limit)


@dataclass(frozen=True)
class DatasetPreview:
    """One bounded accepted-text preview returned to the local caller only."""

    source_ordinal: int
    rendered_identity: ContentIdentity
    split: str
    text: str
    token_count: int

    def __post_init__(self) -> None:
        require_positive_int("source_ordinal", self.source_ordinal)
        if not isinstance(self.rendered_identity, ContentIdentity):
            raise RecordValidationError("rendered_identity must be a content identity")
        require_identifier("split", self.split)
        if not isinstance(self.text, str) or not self.text:
            raise RecordValidationError("preview text must not be empty")
        if rendered_example_identity(self.text) != self.rendered_identity:
            raise RecordValidationError("preview rendered identity mismatch")
        require_non_negative_int("token_count", self.token_count)

    @property
    def selection(self) -> PreviewSelection:
        """Return the value-free persisted projection of this private preview."""

        return PreviewSelection(
            self.source_ordinal,
            self.rendered_identity,
            self.split,
            self.token_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_ordinal": self.source_ordinal,
            "rendered_identity": identity_fields(self.rendered_identity),
            "split": self.split,
            "text": self.text,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class PreparedDataset:
    """Persisted immutable evidence plus exact local training bytes and previews."""

    version: DatasetVersion
    rendered_bytes: bytes
    previews: tuple[DatasetPreview, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, DatasetVersion):
            raise RecordValidationError("version must be a DatasetVersion")
        if not isinstance(self.rendered_bytes, bytes):
            raise RecordValidationError("rendered_bytes must be bytes")
        if _bytes_identity(self.rendered_bytes) != self.version.rendered_bytes_identity:
            raise RecordValidationError("rendered bytes identity mismatch")
        if len(self.rendered_bytes) != self.version.rendered_bytes_count:
            raise RecordValidationError("rendered bytes count mismatch")
        if not isinstance(self.previews, tuple) or any(
            not isinstance(item, DatasetPreview) for item in self.previews
        ):
            raise RecordValidationError("previews contain an invalid value")
        if tuple(item.selection for item in self.previews) != (
            self.version.preview_selections
        ):
            raise RecordValidationError(
                "private previews do not match persisted preview selections"
            )


@dataclass(frozen=True)
class ExclusionReasonCount:
    phase: ExclusionPhase
    reason_code: str
    count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "reason_code": self.reason_code,
            "count": self.count,
        }


@dataclass(frozen=True)
class CorrectionReport:
    """Value-free guidance for correcting the authoritative source externally."""

    dataset_version_identity: ContentIdentity
    accepted_rows: int
    exclusions: tuple[ExclusionReceipt, ...]
    reason_counts: tuple[ExclusionReasonCount, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_version_identity": identity_fields(self.dataset_version_identity),
            "accepted_rows": self.accepted_rows,
            "exclusions": [item.to_dict() for item in self.exclusions],
            "reason_counts": [item.to_dict() for item in self.reason_counts],
        }


@dataclass(frozen=True)
class SplitChange:
    """A shared accepted content identity whose deterministic split changed."""

    rendered_identity: ContentIdentity
    previous_split: str
    current_split: str

    def __post_init__(self) -> None:
        if not isinstance(self.rendered_identity, ContentIdentity):
            raise RecordValidationError("rendered_identity must be a content identity")
        require_identifier("previous_split", self.previous_split)
        require_identifier("current_split", self.current_split)
        if self.previous_split == self.current_split:
            raise RecordValidationError("split change must contain different values")

    def to_dict(self) -> dict[str, object]:
        return {
            "rendered_identity": identity_fields(self.rendered_identity),
            "previous_split": self.previous_split,
            "current_split": self.current_split,
        }


@dataclass(frozen=True)
class ReimportValueDelta:
    """One deeply immutable, exact JSON before/after value change."""

    field: str
    before: object
    after: object

    def __post_init__(self) -> None:
        require_identifier("delta field", self.field)
        before = freeze_json_value(self.before, field=f"{self.field} before")
        after = freeze_json_value(self.after, field=f"{self.field} after")
        if before == after:
            raise RecordValidationError("reimport delta values must differ")
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "before": thaw_json(self.before),
            "after": thaw_json(self.after),
        }


@dataclass(frozen=True)
class SharedTokenCountDelta:
    """Token-count change for content accepted by both immutable versions."""

    rendered_identity: ContentIdentity
    before_token_count: int
    after_token_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.rendered_identity, ContentIdentity):
            raise RecordValidationError("rendered_identity must be a content identity")
        require_non_negative_int("before_token_count", self.before_token_count)
        require_non_negative_int("after_token_count", self.after_token_count)
        if self.before_token_count == self.after_token_count:
            raise RecordValidationError("token-count delta values must differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "rendered_identity": identity_fields(self.rendered_identity),
            "before_token_count": self.before_token_count,
            "after_token_count": self.after_token_count,
        }


@dataclass(frozen=True)
class ReimportComparison:
    """Immutable-version comparison without rewriting either source version."""

    previous_version_identity: ContentIdentity
    current_version_identity: ContentIdentity
    added_content: tuple[ContentIdentity, ...]
    removed_content: tuple[ContentIdentity, ...]
    split_changes: tuple[SplitChange, ...]
    governed_configuration_deltas: tuple[ReimportValueDelta, ...]
    summary_statistics_deltas: tuple[ReimportValueDelta, ...]
    shared_token_count_deltas: tuple[SharedTokenCountDelta, ...]
    previous_exclusions: tuple[ExclusionReceipt, ...]
    current_exclusions: tuple[ExclusionReceipt, ...]

    def __post_init__(self) -> None:
        for name in ("previous_version_identity", "current_version_identity"):
            if not isinstance(getattr(self, name), ContentIdentity):
                raise RecordValidationError(f"{name} must be a content identity")
        for name in ("added_content", "removed_content"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or any(not isinstance(item, ContentIdentity) for item in values)
                or values != tuple(sorted(set(values), key=lambda item: item.value))
            ):
                raise RecordValidationError(f"{name} order is not canonical")
        if set(self.added_content) & set(self.removed_content):
            raise RecordValidationError("added and removed content must be disjoint")
        non_shared_content = set(self.added_content) | set(self.removed_content)
        if (
            not isinstance(self.split_changes, tuple)
            or any(not isinstance(item, SplitChange) for item in self.split_changes)
            or len({item.rendered_identity for item in self.split_changes})
            != len(self.split_changes)
            or self.split_changes
            != tuple(
                sorted(
                    self.split_changes,
                    key=lambda item: item.rendered_identity.value,
                )
            )
        ):
            raise RecordValidationError("split changes order is not canonical")
        if any(
            item.rendered_identity in non_shared_content for item in self.split_changes
        ):
            raise RecordValidationError("split changes require shared content")
        _require_canonical_value_deltas(
            self.governed_configuration_deltas,
            _GOVERNED_CONFIGURATION_FIELDS,
            field="governed_configuration_deltas",
        )
        _require_canonical_value_deltas(
            self.summary_statistics_deltas,
            _SUMMARY_STATISTICS_FIELDS,
            field="summary_statistics_deltas",
        )
        if (
            not isinstance(self.shared_token_count_deltas, tuple)
            or any(
                not isinstance(item, SharedTokenCountDelta)
                for item in self.shared_token_count_deltas
            )
            or len({item.rendered_identity for item in self.shared_token_count_deltas})
            != len(self.shared_token_count_deltas)
            or self.shared_token_count_deltas
            != tuple(
                sorted(
                    self.shared_token_count_deltas,
                    key=lambda item: item.rendered_identity.value,
                )
            )
        ):
            raise RecordValidationError("token-count deltas order is not canonical")
        if any(
            item.rendered_identity in non_shared_content
            for item in self.shared_token_count_deltas
        ):
            raise RecordValidationError("token-count deltas require shared content")
        for name in ("previous_exclusions", "current_exclusions"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(item, ExclusionReceipt) for item in values
            ):
                raise RecordValidationError(f"{name} contains an invalid value")

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_version_identity": identity_fields(
                self.previous_version_identity
            ),
            "current_version_identity": identity_fields(self.current_version_identity),
            "added_content": [identity_fields(item) for item in self.added_content],
            "removed_content": [identity_fields(item) for item in self.removed_content],
            "split_changes": [item.to_dict() for item in self.split_changes],
            "governed_configuration_deltas": [
                item.to_dict() for item in self.governed_configuration_deltas
            ],
            "summary_statistics_deltas": [
                item.to_dict() for item in self.summary_statistics_deltas
            ],
            "shared_token_count_deltas": [
                item.to_dict() for item in self.shared_token_count_deltas
            ],
            "previous_exclusions": [
                item.to_dict() for item in self.previous_exclusions
            ],
            "current_exclusions": [item.to_dict() for item in self.current_exclusions],
        }


def _require_canonical_value_deltas(
    deltas: tuple[ReimportValueDelta, ...],
    field_order: tuple[str, ...],
    *,
    field: str,
) -> None:
    if not isinstance(deltas, tuple) or any(
        not isinstance(item, ReimportValueDelta) for item in deltas
    ):
        raise RecordValidationError(f"{field} contains an invalid value")
    positions = {name: index for index, name in enumerate(field_order)}
    names = tuple(item.field for item in deltas)
    if (
        len(set(names)) != len(names)
        or any(name not in positions for name in names)
        or names != tuple(sorted(names, key=positions.__getitem__))
    ):
        raise RecordValidationError(f"{field} order is not canonical")


def _changed_value_deltas(
    before: Mapping[str, object],
    after: Mapping[str, object],
    field_order: tuple[str, ...],
) -> tuple[ReimportValueDelta, ...]:
    if set(before) != set(field_order) or set(after) != set(field_order):
        raise ApplicationServiceError("dataset_comparison_invariant_failed")
    return tuple(
        ReimportValueDelta(field, before[field], after[field])
        for field in field_order
        if before[field] != after[field]
    )


@dataclass(frozen=True)
class _RenderedCandidate:
    source_ordinal: int
    identity: ContentIdentity
    text: str
    token_count: int
    split: str


class DatasetService:
    """Prepare and persist deterministic dataset versions without network access."""

    def __init__(self, project_root: Path | str) -> None:
        self.store = TypedEvidenceStore(project_root)

    def import_json(
        self, source_bytes: bytes, request: DatasetImportRequest
    ) -> PreparedDataset:
        return self.import_source(JsonDatasetAdapter().load(source_bytes), request)

    def import_jsonl(
        self, source_bytes: bytes, request: DatasetImportRequest
    ) -> PreparedDataset:
        return self.import_source(JsonlDatasetAdapter().load(source_bytes), request)

    def import_csv(
        self, source_bytes: bytes, request: DatasetImportRequest
    ) -> PreparedDataset:
        return self.import_source(CsvDatasetAdapter().load(source_bytes), request)

    def import_hugging_face_rows(
        self,
        rows: Sequence[Mapping[str, object]],
        request: DatasetImportRequest,
    ) -> PreparedDataset:
        return self.import_source(HuggingFaceRowsDatasetAdapter().load(rows), request)

    def import_source(
        self, source: ImportedSource, request: DatasetImportRequest
    ) -> PreparedDataset:
        """Run the complete pipeline, persist the record, and return exact bytes."""

        if not isinstance(source, ImportedSource):
            raise ApplicationServiceError("dataset_source_invalid")
        if not isinstance(request, DatasetImportRequest):
            raise ApplicationServiceError("dataset_request_invalid")
        descriptor, rows = _verified_imported_source(source)
        tokenizer_identity = _tokenizer_identity(request.tokenizer)
        candidates, exclusions = _prepare_candidates(rows, request)
        if _tokenizer_identity(request.tokenizer) != tokenizer_identity:
            raise ApplicationServiceError("tokenizer_nondeterministic")
        if not candidates:
            raise ApplicationServiceError("dataset_has_no_accepted_rows")
        membership = tuple(
            sorted(
                (
                    SplitMembership(candidate.identity, candidate.split)
                    for candidate in candidates
                ),
                key=lambda item: (item.rendered_identity.value, item.split),
            )
        )
        rendered_bytes = b"".join(
            dumps_canonical_json(
                {
                    "rendered_identity": identity_fields(candidate.identity),
                    "source_ordinal": candidate.source_ordinal,
                    "split": candidate.split,
                    "text": candidate.text,
                }
            )
            for candidate in candidates
        )
        accepted = tuple(
            AcceptedExample(
                candidate.source_ordinal,
                candidate.identity,
                len(candidate.text.encode("utf-8")),
                candidate.token_count,
            )
            for candidate in candidates
        )
        statistics = _statistics(
            descriptor.row_count,
            accepted,
            exclusions,
            request.split_rule.parts,
            membership,
        )
        try:
            previews = tuple(
                DatasetPreview(
                    candidate.source_ordinal,
                    candidate.identity,
                    candidate.split,
                    candidate.text,
                    candidate.token_count,
                )
                for candidate in candidates[: request.preview_limit]
            )
            preview_selections = tuple(item.selection for item in previews)
            version = DatasetVersion(
                version_id=request.version_id,
                source=descriptor,
                field_mapping=request.field_mapping,
                renderer=request.renderer,
                renderer_identity=renderer_identity(
                    request.field_mapping, request.renderer
                ),
                filter_rule=request.filter_rule,
                deduplication_rule=request.deduplication_rule,
                tokenizer_identity=tokenizer_identity,
                split_rule=request.split_rule,
                split_identity=split_membership_identity(
                    request.split_rule, membership
                ),
                rendered_bytes_format=RENDERED_BYTES_FORMAT,
                rendered_bytes_identity=_bytes_identity(rendered_bytes),
                rendered_bytes_count=len(rendered_bytes),
                preview_limit=request.preview_limit,
                preview_selections=preview_selections,
                accepted_examples=accepted,
                split_membership=membership,
                exclusions=exclusions,
                statistics=statistics,
            )
            require_no_conflicting_logical_revision(
                self.store,
                version,
                conflict_code="dataset_version_conflict",
            )
            write_record_idempotently(
                self.store,
                version,
                conflict_code="dataset_version_conflict",
            )
            self.store.verify()
        except ApplicationServiceError:
            raise
        except (EvidenceError, RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError("dataset_persistence_failed") from None
        return PreparedDataset(version, rendered_bytes, previews)

    @staticmethod
    def correction_report(version: DatasetVersion) -> CorrectionReport:
        if not isinstance(version, DatasetVersion):
            raise ApplicationServiceError("dataset_version_invalid")
        counts = Counter((item.phase, item.reason_code) for item in version.exclusions)
        reason_counts = tuple(
            ExclusionReasonCount(phase, reason, counts[(phase, reason)])
            for phase, reason in sorted(
                counts, key=lambda item: (item[0].value, item[1])
            )
        )
        return CorrectionReport(
            version.identity,
            len(version.accepted_examples),
            version.exclusions,
            reason_counts,
        )

    @staticmethod
    def compare_reimport(
        previous: DatasetVersion, current: DatasetVersion
    ) -> ReimportComparison:
        if not isinstance(previous, DatasetVersion) or not isinstance(
            current, DatasetVersion
        ):
            raise ApplicationServiceError("dataset_version_invalid")
        previous_splits = {
            item.rendered_identity: item.split for item in previous.split_membership
        }
        current_splits = {
            item.rendered_identity: item.split for item in current.split_membership
        }
        previous_ids = set(previous_splits)
        current_ids = set(current_splits)
        added = tuple(sorted(current_ids - previous_ids, key=lambda item: item.value))
        removed = tuple(sorted(previous_ids - current_ids, key=lambda item: item.value))
        split_changes = tuple(
            SplitChange(identity, previous_splits[identity], current_splits[identity])
            for identity in sorted(
                previous_ids & current_ids, key=lambda item: item.value
            )
            if previous_splits[identity] != current_splits[identity]
        )
        previous_configuration: dict[str, object] = {
            "source.adapter": previous.source.adapter.value,
            "field_mapping": previous.field_mapping.to_dict(),
            "renderer": previous.renderer.to_dict(),
            "filter_rule": previous.filter_rule.to_dict(),
            "deduplication_rule": previous.deduplication_rule.to_dict(),
            "tokenizer_identity": identity_fields(previous.tokenizer_identity),
            "split_rule": previous.split_rule.to_dict(),
            "preview_limit": previous.preview_limit,
        }
        current_configuration: dict[str, object] = {
            "source.adapter": current.source.adapter.value,
            "field_mapping": current.field_mapping.to_dict(),
            "renderer": current.renderer.to_dict(),
            "filter_rule": current.filter_rule.to_dict(),
            "deduplication_rule": current.deduplication_rule.to_dict(),
            "tokenizer_identity": identity_fields(current.tokenizer_identity),
            "split_rule": current.split_rule.to_dict(),
            "preview_limit": current.preview_limit,
        }
        previous_statistics = previous.statistics.to_dict()
        current_statistics = current.statistics.to_dict()
        previous_tokens = {
            item.rendered_identity: item.token_count
            for item in previous.accepted_examples
        }
        current_tokens = {
            item.rendered_identity: item.token_count
            for item in current.accepted_examples
        }
        shared_token_count_deltas = tuple(
            SharedTokenCountDelta(
                identity,
                previous_tokens[identity],
                current_tokens[identity],
            )
            for identity in sorted(
                previous_ids & current_ids,
                key=lambda item: item.value,
            )
            if previous_tokens[identity] != current_tokens[identity]
        )
        return ReimportComparison(
            previous.identity,
            current.identity,
            added,
            removed,
            split_changes,
            _changed_value_deltas(
                previous_configuration,
                current_configuration,
                _GOVERNED_CONFIGURATION_FIELDS,
            ),
            _changed_value_deltas(
                previous_statistics,
                current_statistics,
                _SUMMARY_STATISTICS_FIELDS,
            ),
            shared_token_count_deltas,
            previous.exclusions,
            current.exclusions,
        )


def _prepare_candidates(
    rows: tuple[Mapping[str, object], ...], request: DatasetImportRequest
) -> tuple[tuple[_RenderedCandidate, ...], tuple[ExclusionReceipt, ...]]:
    candidates: list[_RenderedCandidate] = []
    exclusions: list[ExclusionReceipt] = []
    retained_by_text: dict[str, int] = {}
    for source_ordinal, row in enumerate(rows, 1):
        mapped, reason = _map_row(row, request.field_mapping)
        if reason is not None:
            exclusions.append(
                ExclusionReceipt(
                    source_ordinal,
                    ExclusionPhase.VALIDATION,
                    reason,
                )
            )
            continue
        if mapped is None:
            raise ApplicationServiceError("dataset_pipeline_invariant_failed")
        text = _render(mapped, request.renderer)
        character_count = len(text)
        if character_count < request.filter_rule.minimum_characters:
            exclusions.append(
                ExclusionReceipt(
                    source_ordinal,
                    ExclusionPhase.FILTERING,
                    "below_minimum_characters",
                )
            )
            continue
        if (
            request.filter_rule.maximum_characters is not None
            and character_count > request.filter_rule.maximum_characters
        ):
            exclusions.append(
                ExclusionReceipt(
                    source_ordinal,
                    ExclusionPhase.FILTERING,
                    "above_maximum_characters",
                )
            )
            continue
        token_count = _count_tokens(request.tokenizer, text)
        if (
            request.filter_rule.maximum_tokens is not None
            and token_count > request.filter_rule.maximum_tokens
        ):
            exclusions.append(
                ExclusionReceipt(
                    source_ordinal,
                    ExclusionPhase.FILTERING,
                    "above_maximum_tokens",
                )
            )
            continue
        retained = retained_by_text.get(text)
        if retained is not None:
            exclusions.append(
                ExclusionReceipt(
                    source_ordinal,
                    ExclusionPhase.DEDUPLICATION,
                    "duplicate_rendered_text",
                    retained,
                )
            )
            continue
        retained_by_text[text] = source_ordinal
        identity = rendered_example_identity(text)
        candidates.append(
            _RenderedCandidate(
                source_ordinal,
                identity,
                text,
                token_count,
                _assign_split(identity, request.split_rule),
            )
        )
    return tuple(candidates), tuple(exclusions)


def _map_row(
    row: Mapping[str, object], mapping: FieldMapping
) -> tuple[tuple[str, str | None, str] | None, str | None]:
    required = (mapping.instruction_field, mapping.response_field)
    optional = (mapping.context_field,) if mapping.context_field is not None else ()
    if any(field not in row for field in (*required, *optional)):
        return None, "missing_mapped_field"
    instruction = row[mapping.instruction_field]
    response = row[mapping.response_field]
    context = row[mapping.context_field] if mapping.context_field is not None else None
    if not isinstance(instruction, str) or not isinstance(response, str):
        return None, "mapped_value_not_text"
    if context is not None and not isinstance(context, str):
        return None, "mapped_value_not_text"
    if not instruction.strip() or not response.strip():
        return None, "required_text_empty"
    return (instruction, context, response), None


def _render(mapped: tuple[str, str | None, str], renderer: RendererSpec) -> str:
    if (
        renderer.kind is not RendererKind.INSTRUCTION_RESPONSE
        or renderer.version != "v1"
    ):
        raise ApplicationServiceError("renderer_unsupported")
    instruction, context, response = mapped
    sections = ["### Instruction", instruction]
    if context:
        sections.extend(("### Context", context))
    sections.extend(("### Response", response))
    return "\n".join(sections)


def _tokenizer_identity(tokenizer: DeterministicTokenizer) -> ContentIdentity:
    try:
        identity = tokenizer.identity
    except Exception:
        raise ApplicationServiceError("tokenizer_identity_invalid") from None
    if not isinstance(identity, ContentIdentity):
        raise ApplicationServiceError("tokenizer_identity_invalid")
    return identity


def _count_tokens(tokenizer: DeterministicTokenizer, text: str) -> int:
    try:
        first = tokenizer.count_tokens(text)
        second = tokenizer.count_tokens(text)
    except Exception:
        raise ApplicationServiceError("tokenizer_failed") from None
    if (
        isinstance(first, bool)
        or not isinstance(first, int)
        or first < 0
        or isinstance(second, bool)
        or not isinstance(second, int)
        or second < 0
        or first != second
    ):
        raise ApplicationServiceError("tokenizer_nondeterministic")
    return first


def _assign_split(identity: ContentIdentity, rule: SplitRule) -> str:
    preimage = _SPLIT_BUCKET_PREFIX + dumps_canonical_json(
        {
            "algorithm": rule.algorithm,
            "seed": rule.seed,
            "rendered_identity": identity_fields(identity),
        }
    )
    bucket = int(hashlib.sha256(preimage).hexdigest(), 16) % sum(
        part.weight for part in rule.parts
    )
    boundary = 0
    for part in rule.parts:
        boundary += part.weight
        if bucket < boundary:
            return part.name
    raise ApplicationServiceError("split_assignment_failed")


def _statistics(
    source_rows: int,
    accepted: tuple[AcceptedExample, ...],
    exclusions: tuple[ExclusionReceipt, ...],
    parts: tuple[SplitPart, ...],
    membership: tuple[SplitMembership, ...],
) -> DatasetStatistics:
    tokens = tuple(item.token_count for item in accepted)
    return DatasetStatistics(
        source_rows=source_rows,
        accepted_rows=len(accepted),
        excluded_rows=len(exclusions),
        duplicate_rows=sum(
            item.phase is ExclusionPhase.DEDUPLICATION for item in exclusions
        ),
        total_tokens=sum(tokens),
        minimum_tokens=min(tokens),
        maximum_tokens=max(tokens),
        split_counts=tuple(
            SplitCount(part.name, sum(item.split == part.name for item in membership))
            for part in parts
        ),
    )


def _require_source_bytes(source: object) -> bytes:
    if not isinstance(source, bytes):
        raise DatasetAdapterError("source_bytes_required")
    return bytes(source)


def _parse_json(data: bytes, *, failure_code: str) -> object:
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        raise DatasetAdapterError("source_not_utf8") from None
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except DatasetAdapterError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError):
        raise DatasetAdapterError(failure_code) from None


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetAdapterError("json_duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise DatasetAdapterError("json_non_finite_number")


def _freeze_rows(
    values: Sequence[object], *, failure_code: str
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for value in values:
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) for key in value
        ):
            raise DatasetAdapterError(failure_code)
        try:
            encoded = _portable_json_bytes(_thaw_source_value(value))
            copied = json.loads(encoded)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            raise DatasetAdapterError(failure_code) from None
        if not isinstance(copied, dict):
            raise DatasetAdapterError(failure_code)
        frozen = _freeze_source_value(copied)
        if not isinstance(frozen, Mapping):
            raise DatasetAdapterError(failure_code)
        rows.append(frozen)
    return tuple(rows)


def _freeze_source_value(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_source_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_source_value(item) for item in value)
    return value


def _thaw_source_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_source_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_source_value(item) for item in value]
    return value


def _portable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _bytes_identity(value: bytes) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(value).hexdigest())


def _verified_imported_source(
    source: ImportedSource,
) -> tuple[SourceDescriptor, tuple[Mapping[str, object], ...]]:
    """Copy and verify imported rows against their exact identity preimage."""

    try:
        descriptor = source.descriptor
        preimage = source._identity_preimage
        if not isinstance(descriptor, SourceDescriptor) or not isinstance(
            preimage, bytes
        ):
            raise DatasetAdapterError("adapter_result_invalid")
        rows = _freeze_rows(source.rows, failure_code="adapter_result_invalid")
        if (
            len(rows) != descriptor.row_count
            or _bytes_identity(preimage) != descriptor.source_identity
        ):
            raise DatasetAdapterError("adapter_result_invalid")
        expected_rows = _rows_from_identity_preimage(descriptor.adapter, preimage)
        if rows != expected_rows:
            raise DatasetAdapterError("adapter_result_invalid")
    except Exception:
        raise ApplicationServiceError("dataset_source_invalid") from None
    return descriptor, rows


def _rows_from_identity_preimage(
    adapter: DatasetAdapter,
    preimage: bytes,
) -> tuple[Mapping[str, object], ...]:
    if adapter is DatasetAdapter.JSON:
        return JsonDatasetAdapter().load(preimage).rows
    if adapter is DatasetAdapter.JSONL:
        return JsonlDatasetAdapter().load(preimage).rows
    if adapter is DatasetAdapter.CSV:
        return CsvDatasetAdapter().load(preimage).rows
    if adapter is DatasetAdapter.HUGGING_FACE_ROWS:
        value = _parse_json(preimage, failure_code="hugging_face_rows_invalid")
        if not isinstance(value, list):
            raise DatasetAdapterError("hugging_face_rows_invalid")
        rows = _freeze_rows(value, failure_code="hugging_face_rows_invalid")
        if _portable_json_bytes([_thaw_source_value(row) for row in rows]) != preimage:
            raise DatasetAdapterError("hugging_face_rows_invalid")
        return rows
    raise DatasetAdapterError("adapter_result_invalid")


def _imported_bytes_source(
    adapter: DatasetAdapter,
    source_bytes: bytes,
    rows: tuple[Mapping[str, object], ...],
) -> ImportedSource:
    return ImportedSource(
        SourceDescriptor(adapter, _bytes_identity(source_bytes), len(rows)),
        rows,
        source_bytes,
    )
