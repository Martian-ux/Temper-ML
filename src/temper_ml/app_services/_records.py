"""Shared idempotent canonical-record persistence for application services."""

from __future__ import annotations

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.domain.records import (
    CORE_LOGICAL_ID_FIELDS,
    TypedRecord,
    record_reference,
)
from temper_ml.store.evidence import (
    EvidenceExists,
    StoredTypedRecord,
    TypedEvidenceStore,
)


def write_record_idempotently(
    store: TypedEvidenceStore,
    record: TypedRecord,
    *,
    conflict_code: str,
) -> StoredTypedRecord:
    """Write one exact record or recover an interrupted identical write."""

    matches = tuple(
        stored
        for stored in store.iter_records()
        if stored.envelope.record_type == record.RECORD_TYPE
        and stored.envelope.identity == record.identity
    )
    if len(matches) > 1:
        raise ApplicationServiceError(conflict_code)
    if matches:
        if matches[0].envelope.to_dict() != record.to_dict():
            raise ApplicationServiceError(conflict_code)
        return matches[0]
    try:
        return store.write_record(record)
    except EvidenceExists:
        stored = store.read_record(record_reference(record))
        if stored.envelope.to_dict() != record.to_dict():
            raise ApplicationServiceError(conflict_code) from None
        return stored


def require_no_conflicting_logical_revision(
    store: TypedEvidenceStore,
    record: TypedRecord,
    *,
    conflict_code: str,
) -> None:
    """Enforce a single revision only for operations whose contract requires it."""

    logical_field = CORE_LOGICAL_ID_FIELDS[record.RECORD_TYPE]
    logical_id = getattr(record, logical_field)
    if any(
        stored.envelope.record_type == record.RECORD_TYPE
        and stored.reference.logical_id == logical_id
        and stored.envelope.identity != record.identity
        for stored in store.iter_records()
    ):
        raise ApplicationServiceError(conflict_code)
