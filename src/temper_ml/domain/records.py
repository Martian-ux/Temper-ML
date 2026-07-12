"""Typed, immutable envelopes for Temper canonical domain records."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import MISSING as DATACLASS_MISSING
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from enum import Enum
import re
from types import MappingProxyType
from types import UnionType
from typing import (
    Any,
    ClassVar,
    Protocol,
    Self,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    ProjectionError,
    ProjectionRegistration,
    ProjectionRegistry,
    content_identity,
)
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)

SCHEMA_VERSION = "v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RECORD_TYPE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


class RecordValidationError(ValueError):
    """Raised when a domain record cannot be canonical or immutable."""


_CORE_RECORD_TYPES = (
    "adapter_export",
    "artifact",
    "artifact_availability",
    "baseline_policy",
    "base_model_revision",
    "compatibility_group",
    "execution_target",
    "experiment",
    "experiment_derivation",
    "hardware_capability_profile",
    "hardware_requirements",
    "local_use_session",
    "manifest_diff",
    "project",
    "project_policy",
    "recipe",
    "recipe_resolution",
    "run",
    "task_definition",
)

CORE_LOGICAL_ID_FIELDS = MappingProxyType(
    {
        "adapter_export": "export_id",
        "artifact": "artifact_id",
        "artifact_availability": "availability_id",
        "baseline_policy": "policy_id",
        "base_model_revision": "model_id",
        "compatibility_group": "group_id",
        "execution_target": "target_id",
        "experiment": "experiment_id",
        "experiment_derivation": "derivation_id",
        "hardware_capability_profile": "profile_id",
        "hardware_requirements": "requirements_id",
        "local_use_session": "session_id",
        "manifest_diff": "diff_id",
        "project": "project_id",
        "project_policy": "policy_id",
        "recipe": "recipe_id",
        "recipe_resolution": "resolution_id",
        "run": "run_id",
        "task_definition": "task_id",
    }
)

CORE_PROJECTION_REGISTRY = ProjectionRegistry(
    ProjectionRegistration(
        record_type=record_type,
        schema_version=SCHEMA_VERSION,
        projection=HashProjection(f"record.{record_type}", "v1"),
    )
    for record_type in _CORE_RECORD_TYPES
)

_RECORD_CLASSES: dict[str, type[Any]] = {}


class FrozenJsonObject(Mapping[str, Any]):
    """A recursively frozen JSON object with mapping value semantics."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = MappingProxyType(dict(data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def freeze_json_object(value: Mapping[str, Any], *, field: str) -> FrozenJsonObject:
    """Validate, canonical-copy, and deeply freeze one JSON object."""

    if not isinstance(value, Mapping):
        raise RecordValidationError(f"{field} must be a JSON object")
    copied = _copy_json_input(value, field=field)
    try:
        canonical = loads_canonical_json(dumps_canonical_json(copied))
    except CanonicalJsonError as exc:
        raise RecordValidationError(f"{field} is not canonical JSON") from exc
    if not isinstance(canonical, dict):
        raise RecordValidationError(f"{field} must be a JSON object")
    frozen = _freeze_json(canonical)
    if not isinstance(frozen, FrozenJsonObject):
        raise RecordValidationError(f"{field} must be a JSON object")
    return frozen


def freeze_json_value(value: Any, *, field: str) -> Any:
    """Validate, canonical-copy, and deeply freeze any JSON value."""

    copied = _copy_json_input(value, field=field)
    try:
        canonical = loads_canonical_json(dumps_canonical_json(copied))
    except CanonicalJsonError as exc:
        raise RecordValidationError(f"{field} is not canonical JSON") from exc
    return _freeze_json(canonical)


def thaw_json(value: Any) -> Any:
    """Return ordinary JSON containers from recursively frozen JSON."""

    if isinstance(value, FrozenJsonObject):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def require_identifier(field: str, value: str) -> str:
    """Validate a bounded, portable logical identifier."""

    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise RecordValidationError(f"{field} is not a valid logical identifier")
    return value


def require_text(field: str, value: str) -> str:
    """Validate user-visible canonical text without hidden surrounding space."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise RecordValidationError(f"{field} must be non-empty canonical text")
    return value


def require_positive_int(field: str, value: int) -> int:
    """Validate a positive integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecordValidationError(f"{field} must be a positive integer")
    return value


def require_non_negative_int(field: str, value: int) -> int:
    """Validate a non-negative integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecordValidationError(f"{field} must be a non-negative integer")
    return value


def require_string_tuple(
    field: str,
    value: Sequence[str],
    *,
    non_empty: bool = True,
    sorted_values: bool = False,
) -> tuple[str, ...]:
    """Validate an immutable sequence of unique non-empty strings."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecordValidationError(f"{field} must be a sequence of strings")
    items = tuple(value)
    if non_empty and not items:
        raise RecordValidationError(f"{field} must not be empty")
    for item in items:
        require_text(field, item)
    if len(set(items)) != len(items):
        raise RecordValidationError(f"{field} must not contain duplicates")
    return tuple(sorted(items)) if sorted_values else items


def identity_fields(identity: ContentIdentity) -> dict[str, str]:
    """Project a content identity into canonical JSON fields."""

    if not isinstance(identity, ContentIdentity):
        raise RecordValidationError("expected a ContentIdentity")
    return {"algorithm": identity.algorithm, "value": identity.value}


def parse_identity(value: Mapping[str, Any], *, field: str) -> ContentIdentity:
    """Parse the strict JSON representation of a content identity."""

    if not isinstance(value, Mapping) or set(value) != {"algorithm", "value"}:
        raise RecordValidationError(f"{field} is not a content identity")
    algorithm = value["algorithm"]
    digest = value["value"]
    if not isinstance(algorithm, str) or not isinstance(digest, str):
        raise RecordValidationError(f"{field} is not a content identity")
    try:
        return ContentIdentity(algorithm, digest)
    except ProjectionError as exc:
        raise RecordValidationError(f"{field} is not a content identity") from exc


@dataclass(frozen=True)
class RecordReference:
    """A logical record identifier pinned to one immutable revision."""

    record_type: str
    logical_id: str
    identity: ContentIdentity

    def __post_init__(self) -> None:
        if (
            not isinstance(self.record_type, str)
            or _RECORD_TYPE.fullmatch(self.record_type) is None
        ):
            raise RecordValidationError("record_type is invalid")
        require_identifier("logical_id", self.logical_id)
        identity_fields(self.identity)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "logical_id": self.logical_id,
            "identity": identity_fields(self.identity),
        }


class RecordContract(Protocol):
    """Structural contract implemented by every typed Slice 2 record."""

    RECORD_TYPE: ClassVar[str]
    SCHEMA_VERSION: ClassVar[str]

    def to_payload(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RecordEnvelope:
    """Canonical typed envelope whose payload is covered by its identity."""

    record_type: str
    schema_version: str
    projection: HashProjection
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.record_type, str)
            or _RECORD_TYPE.fullmatch(self.record_type) is None
        ):
            raise RecordValidationError("record_type is invalid")
        if not isinstance(self.schema_version, str):
            raise RecordValidationError("schema_version is invalid")
        registration = CORE_PROJECTION_REGISTRY.resolve(
            self.record_type, self.schema_version
        )
        if registration.projection != self.projection:
            raise RecordValidationError("record projection does not match registry")
        object.__setattr__(
            self,
            "payload",
            freeze_json_object(self.payload, field="payload"),
        )
        decoded = _decode_typed_record(self.record_type, self.payload)
        if dumps_canonical_json(decoded.to_payload()) != dumps_canonical_json(
            thaw_json(self.payload)
        ):
            raise RecordValidationError(
                "record payload is not the canonical typed representation"
            )

    @property
    def projection_version(self) -> str:
        return self.projection.version

    def projected_fields(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "projection_version": self.projection.version,
            "payload": thaw_json(self.payload),
        }

    @property
    def identity(self) -> ContentIdentity:
        return content_identity(self.projection, self.projected_fields())

    def to_dict(self) -> dict[str, object]:
        fields = self.projected_fields()
        fields["identity"] = identity_fields(self.identity)
        return fields

    def to_record(self) -> "TypedRecord":
        """Decode the envelope through its exact registered record constructor."""

        return _decode_typed_record(self.record_type, self.payload)

    @classmethod
    def from_record(cls, record: RecordContract) -> "RecordEnvelope":
        expected = _RECORD_CLASSES.get(record.RECORD_TYPE)
        if expected is None or type(record) is not expected:
            raise RecordValidationError(
                "record object does not match the registered concrete type"
            )
        registration = CORE_PROJECTION_REGISTRY.resolve(
            record.RECORD_TYPE, record.SCHEMA_VERSION
        )
        return cls(
            record_type=record.RECORD_TYPE,
            schema_version=record.SCHEMA_VERSION,
            projection=registration.projection,
            payload=record.to_payload(),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecordEnvelope":
        expected = {
            "record_type",
            "schema_version",
            "projection_version",
            "identity",
            "payload",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise RecordValidationError("record envelope fields are invalid")
        record_type = value["record_type"]
        schema_version = value["schema_version"]
        projection_version = value["projection_version"]
        payload = value["payload"]
        claimed_identity = value["identity"]
        if not isinstance(record_type, str) or not isinstance(schema_version, str):
            raise RecordValidationError("record envelope discriminator is invalid")
        registration = CORE_PROJECTION_REGISTRY.resolve(record_type, schema_version)
        if projection_version != registration.projection.version:
            raise RecordValidationError("record projection version is invalid")
        if not isinstance(payload, Mapping):
            raise RecordValidationError("record envelope payload is invalid")
        if not isinstance(claimed_identity, Mapping):
            raise RecordValidationError("record envelope identity is invalid")
        envelope = cls(
            record_type,
            schema_version,
            registration.projection,
            payload,
        )
        if parse_identity(claimed_identity, field="identity") != envelope.identity:
            raise RecordValidationError("record envelope identity mismatch")
        return envelope


class TypedRecord:
    """Convenience base for typed records backed by a registered envelope."""

    RECORD_TYPE: ClassVar[str]
    SCHEMA_VERSION: ClassVar[str] = SCHEMA_VERSION

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        record_type = cls.__dict__.get("RECORD_TYPE")
        if record_type is None:
            return
        if not isinstance(record_type, str) or record_type not in _CORE_RECORD_TYPES:
            raise RecordValidationError(
                f"unregistered typed record class: {record_type!r}"
            )
        logical_id_field = CORE_LOGICAL_ID_FIELDS[record_type]
        if logical_id_field not in cls.__dict__.get("__annotations__", {}):
            raise RecordValidationError(
                f"typed record lacks logical id field: {logical_id_field!r}"
            )
        existing = _RECORD_CLASSES.get(record_type)
        if existing is not None and existing is not cls:
            raise RecordValidationError(
                f"duplicate typed record class: {record_type!r}"
            )
        _RECORD_CLASSES[record_type] = cls

    def to_payload(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Decode strict wire fields and re-run the concrete constructor invariants."""

        decoded = _decode_dataclass(cls, payload, field=cls.RECORD_TYPE)
        if not isinstance(decoded, cls):
            raise RecordValidationError(
                f"decoded payload is not a {cls.RECORD_TYPE} record"
            )
        return decoded

    def to_envelope(self) -> RecordEnvelope:
        return RecordEnvelope.from_record(self)

    @property
    def identity(self) -> ContentIdentity:
        return self.to_envelope().identity

    def projected_fields(self) -> dict[str, object]:
        return self.to_envelope().projected_fields()

    def to_dict(self) -> dict[str, object]:
        return self.to_envelope().to_dict()


RecordT = TypeVar("RecordT", bound=TypedRecord)


def record_reference(record: RecordT, logical_id: str | None = None) -> RecordReference:
    """Pin a typed record's logical identifier to its current identity."""

    field = CORE_LOGICAL_ID_FIELDS[record.RECORD_TYPE]
    expected = getattr(record, field)
    if logical_id is not None and logical_id != expected:
        raise RecordValidationError(
            f"logical_id must match {record.RECORD_TYPE}.{field}"
        )
    return RecordReference(record.RECORD_TYPE, expected, record.identity)


def _decode_typed_record(record_type: str, payload: Mapping[str, Any]) -> TypedRecord:
    record_class = _RECORD_CLASSES.get(record_type)
    if record_class is None or not issubclass(record_class, TypedRecord):
        raise RecordValidationError(
            f"no concrete typed record is registered for {record_type!r}"
        )
    return record_class.from_payload(payload)


def _decode_dataclass(cls: type[Any], value: Any, *, field: str) -> Any:
    if not is_dataclass(cls) or not isinstance(value, Mapping):
        raise RecordValidationError(f"{field} must be an object")
    init_fields = tuple(item for item in fields(cls) if item.init)
    allowed = {item.name for item in init_fields}
    required = {
        item.name
        for item in init_fields
        if item.default is DATACLASS_MISSING
        and item.default_factory is DATACLASS_MISSING
    }
    present = set(value)
    if not required <= present or not present <= allowed:
        raise RecordValidationError(f"{field} fields are invalid")
    hints = get_type_hints(cls)
    decoded = {
        item.name: _decode_value(
            hints[item.name], value[item.name], field=f"{field}.{item.name}"
        )
        for item in init_fields
        if item.name in value
    }
    try:
        return cls(**decoded)
    except RecordValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"{field} is invalid") from exc


def _decode_value(annotation: Any, value: Any, *, field: str) -> Any:
    if annotation is Any:
        return value
    if annotation is ContentIdentity:
        if not isinstance(value, Mapping):
            raise RecordValidationError(f"{field} must be a content identity")
        return parse_identity(value, field=field)
    if annotation is RecordReference:
        return _decode_reference(value, field=field)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise RecordValidationError(f"{field} has an invalid enum value") from exc
    if isinstance(annotation, type) and issubclass(annotation, TypedRecord):
        if not isinstance(value, Mapping):
            raise RecordValidationError(f"{field} must be a record envelope")
        nested = RecordEnvelope.from_dict(value).to_record()
        if type(nested) is not annotation:
            raise RecordValidationError(f"{field} has the wrong record type")
        return nested
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, UnionType):
        return _decode_union(arguments, value, field=field)
    if origin is tuple:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise RecordValidationError(f"{field} must be an array")
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise RecordValidationError(f"{field} has an unsupported tuple schema")
        return tuple(
            _decode_value(arguments[0], item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise RecordValidationError(f"{field} must be an object")
        if arguments and arguments[0] is not str:
            raise RecordValidationError(f"{field} has unsupported key types")
        return {key: thaw_json(item) for key, item in value.items()}
    if annotation is str:
        if not isinstance(value, str):
            raise RecordValidationError(f"{field} must be a string")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise RecordValidationError(f"{field} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RecordValidationError(f"{field} must be an integer")
        return value
    if annotation is Decimal:
        if not isinstance(value, Decimal):
            raise RecordValidationError(f"{field} must be a Decimal")
        return value
    if annotation is type(None):
        if value is not None:
            raise RecordValidationError(f"{field} must be null")
        return None
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _decode_dataclass(annotation, value, field=field)
    raise RecordValidationError(f"{field} has an unsupported wire type")


def _decode_union(arguments: tuple[Any, ...], value: Any, *, field: str) -> Any:
    if isinstance(value, Mapping) and "kind" in value:
        kind = value["kind"]
        discriminators = {
            "PerModelBaseline": "per_model",
            "ProjectChampionBaseline": "project_champion",
            "FixedReferenceBaseline": "fixed_reference",
        }
        for option in arguments:
            if discriminators.get(getattr(option, "__name__", "")) == kind:
                nested = dict(value)
                del nested["kind"]
                return _decode_value(option, nested, field=field)
        raise RecordValidationError(f"{field} has an invalid discriminator")
    failures: list[RecordValidationError] = []
    for option in arguments:
        try:
            return _decode_value(option, value, field=field)
        except RecordValidationError as exc:
            failures.append(exc)
    raise RecordValidationError(f"{field} does not match an allowed type") from (
        failures[-1] if failures else None
    )


def _decode_reference(value: Any, *, field: str) -> RecordReference:
    if not isinstance(value, Mapping) or set(value) != {
        "record_type",
        "logical_id",
        "identity",
    }:
        raise RecordValidationError(f"{field} must be a record reference")
    record_type = value["record_type"]
    logical_id = value["logical_id"]
    identity = value["identity"]
    if (
        not isinstance(record_type, str)
        or not isinstance(logical_id, str)
        or not isinstance(identity, Mapping)
    ):
        raise RecordValidationError(f"{field} must be a record reference")
    return RecordReference(
        record_type,
        logical_id,
        parse_identity(identity, field=f"{field}.identity"),
    )


def _copy_json_input(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, Decimal)):
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecordValidationError(f"{field} keys must be strings")
            copied[key] = _copy_json_input(item, field=field)
        return copied
    if isinstance(value, (list, tuple)):
        return [_copy_json_input(item, field=field) for item in value]
    raise RecordValidationError(
        f"{field} contains unsupported JSON type: {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenJsonObject(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value
