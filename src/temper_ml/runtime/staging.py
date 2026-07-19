"""Content-identified local staging and ingestion across the Windows/WSL boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Mapping

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.records import (
    RecordValidationError,
    identity_fields,
    parse_identity,
    require_identifier,
)
from temper_ml.filesystem import UnsafeFilesystemPath, ensure_safe_directory
from temper_ml.runtime.paths import PortableLocation, PortablePathError
from temper_ml.store.safe_io import SafeIoError, read_stable_bytes, write_once_bytes

TRANSFER_MANIFEST_PROJECTION = HashProjection("runtime.transfer_manifest", "v1")
TRANSFER_RECEIPT_PROJECTION = HashProjection("runtime.transfer_receipt", "v1")


class StagingError(RuntimeError):
    """A stable local transfer failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class TransferDirection(str, Enum):
    HOST_TO_WORKER = "host_to_worker"
    WORKER_TO_HOST = "worker_to_host"


@dataclass(frozen=True)
class TransferMember:
    logical_location: PortableLocation
    role: str
    content_identity: ContentIdentity
    byte_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.logical_location, PortableLocation):
            raise StagingError("transfer_location_invalid")
        try:
            require_identifier("role", self.role)
        except RecordValidationError:
            raise StagingError("transfer_role_invalid") from None
        if not isinstance(self.content_identity, ContentIdentity):
            raise StagingError("transfer_identity_invalid")
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 0
        ):
            raise StagingError("transfer_byte_count_invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_location": self.logical_location.to_dict(),
            "role": self.role,
            "content_identity": identity_fields(self.content_identity),
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TransferMember":
        if not isinstance(value, Mapping) or set(value) != {
            "logical_location",
            "role",
            "content_identity",
            "byte_count",
        }:
            raise StagingError("transfer_member_invalid")
        raw_location = value["logical_location"]
        if not isinstance(raw_location, Mapping) or set(raw_location) != {
            "logical_path"
        }:
            raise StagingError("transfer_member_invalid")
        raw_identity = value["content_identity"]
        if not isinstance(raw_identity, Mapping):
            raise StagingError("transfer_member_invalid")
        try:
            location = PortableLocation(raw_location["logical_path"])  # type: ignore[arg-type]
            identity = parse_identity(
                raw_identity,
                field="content_identity",
            )
            return cls(
                location,
                value["role"],  # type: ignore[arg-type]
                identity,
                value["byte_count"],  # type: ignore[arg-type]
            )
        except (PortablePathError, RecordValidationError, TypeError, ValueError):
            raise StagingError("transfer_member_invalid") from None


@dataclass(frozen=True)
class TransferManifest:
    direction: TransferDirection
    members: tuple[TransferMember, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.direction, TransferDirection):
            raise StagingError("transfer_direction_invalid")
        if not isinstance(self.members, tuple) or not self.members:
            raise StagingError("transfer_members_invalid")
        if any(not isinstance(member, TransferMember) for member in self.members):
            raise StagingError("transfer_members_invalid")
        ordered = tuple(
            sorted(
                self.members,
                key=lambda member: member.logical_location.logical_path,
            )
        )
        if ordered != self.members:
            raise StagingError("transfer_members_not_ordered")
        paths = tuple(member.logical_location.logical_path for member in self.members)
        if len(set(paths)) != len(paths):
            raise StagingError("transfer_member_duplicate")

    @property
    def identity(self) -> ContentIdentity:
        return content_identity(
            TRANSFER_MANIFEST_PROJECTION,
            {
                "schema_version": "v1",
                "direction": self.direction.value,
                "members": [member.to_dict() for member in self.members],
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction.value,
            "members": [member.to_dict() for member in self.members],
            "identity": identity_fields(self.identity),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TransferManifest":
        if not isinstance(value, Mapping) or set(value) != {
            "direction",
            "members",
            "identity",
        }:
            raise StagingError("transfer_manifest_invalid")
        raw_members = value["members"]
        raw_identity = value["identity"]
        if not isinstance(raw_members, list) or not isinstance(raw_identity, Mapping):
            raise StagingError("transfer_manifest_invalid")
        try:
            manifest = cls(
                TransferDirection(value["direction"]),  # type: ignore[arg-type]
                tuple(TransferMember.from_dict(member) for member in raw_members),
            )
            claimed = parse_identity(
                raw_identity,
                field="manifest_identity",
            )
        except (RecordValidationError, TypeError, ValueError):
            raise StagingError("transfer_manifest_invalid") from None
        if manifest.identity != claimed:
            raise StagingError("transfer_manifest_identity_mismatch")
        return manifest


@dataclass(frozen=True)
class TransferReceipt:
    direction: TransferDirection
    manifest_identity: ContentIdentity
    verified_members: tuple[TransferMember, ...]
    complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.direction, TransferDirection):
            raise StagingError("transfer_receipt_direction_invalid")
        if not isinstance(self.manifest_identity, ContentIdentity):
            raise StagingError("transfer_receipt_manifest_invalid")
        if not isinstance(self.verified_members, tuple):
            raise StagingError("transfer_receipt_members_invalid")
        if any(
            not isinstance(member, TransferMember) for member in self.verified_members
        ):
            raise StagingError("transfer_receipt_members_invalid")
        if self.complete is not True:
            raise StagingError("transfer_receipt_state_invalid")
        try:
            verified_manifest = TransferManifest(
                self.direction,
                self.verified_members,
            )
        except StagingError:
            raise StagingError("transfer_receipt_members_invalid") from None
        if verified_manifest.identity != self.manifest_identity:
            raise StagingError("transfer_receipt_manifest_mismatch")

    @property
    def identity(self) -> ContentIdentity:
        return content_identity(
            TRANSFER_RECEIPT_PROJECTION,
            {
                "schema_version": "v1",
                "direction": self.direction.value,
                "manifest_identity": identity_fields(self.manifest_identity),
                "verified_members": [
                    member.to_dict() for member in self.verified_members
                ],
                "complete": self.complete,
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction.value,
            "manifest_identity": identity_fields(self.manifest_identity),
            "verified_members": [member.to_dict() for member in self.verified_members],
            "complete": self.complete,
            "identity": identity_fields(self.identity),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TransferReceipt":
        if not isinstance(value, Mapping) or set(value) != {
            "direction",
            "manifest_identity",
            "verified_members",
            "complete",
            "identity",
        }:
            raise StagingError("transfer_receipt_invalid")
        raw_members = value["verified_members"]
        raw_manifest_identity = value["manifest_identity"]
        raw_identity = value["identity"]
        if (
            not isinstance(raw_members, list)
            or not isinstance(raw_manifest_identity, Mapping)
            or not isinstance(raw_identity, Mapping)
        ):
            raise StagingError("transfer_receipt_invalid")
        try:
            receipt = cls(
                TransferDirection(value["direction"]),  # type: ignore[arg-type]
                parse_identity(
                    raw_manifest_identity,
                    field="manifest_identity",
                ),
                tuple(TransferMember.from_dict(member) for member in raw_members),
                value["complete"],  # type: ignore[arg-type]
            )
            claimed = parse_identity(
                raw_identity,
                field="receipt_identity",
            )
        except (RecordValidationError, TypeError, ValueError):
            raise StagingError("transfer_receipt_invalid") from None
        if receipt.identity != claimed:
            raise StagingError("transfer_receipt_identity_mismatch")
        return receipt


def build_transfer_manifest(
    direction: TransferDirection,
    members: Mapping[PortableLocation, tuple[str, bytes]],
) -> TransferManifest:
    """Describe exact bytes before they cross the local worker boundary."""

    if not isinstance(members, Mapping) or not members:
        raise StagingError("transfer_members_invalid")
    projected = []
    for location, item in members.items():
        if (
            not isinstance(location, PortableLocation)
            or not isinstance(item, tuple)
            or len(item) != 2
        ):
            raise StagingError("transfer_members_invalid")
        role, payload = item
        if not isinstance(payload, bytes):
            raise StagingError("transfer_payload_invalid")
        projected.append(
            TransferMember(
                location,
                role,
                ContentIdentity("sha256", hashlib.sha256(payload).hexdigest()),
                len(payload),
            )
        )
    return TransferManifest(
        direction,
        tuple(
            sorted(
                projected,
                key=lambda member: member.logical_location.logical_path,
            )
        ),
    )


def stage_transfer(
    root: Path | str,
    manifest: TransferManifest,
    payloads: Mapping[PortableLocation, bytes],
) -> TransferReceipt:
    """Write each manifest member once and verify every final byte."""

    if not isinstance(manifest, TransferManifest) or not isinstance(payloads, Mapping):
        raise StagingError("transfer_input_invalid")
    if set(payloads) != {member.logical_location for member in manifest.members}:
        raise StagingError("transfer_payload_members_mismatch")
    base = Path(root)
    try:
        ensure_safe_directory(base)
    except (OSError, UnsafeFilesystemPath):
        raise StagingError("transfer_root_unavailable") from None
    for member in manifest.members:
        payload = payloads[member.logical_location]
        _verify_payload(member, payload)
        destination = _member_path(base, member.logical_location)
        try:
            write_once_bytes(destination, payload)
        except FileExistsError:
            try:
                existing = read_stable_bytes(destination)
            except SafeIoError:
                raise StagingError("transfer_existing_member_unreadable") from None
            if existing != payload:
                raise StagingError("transfer_existing_member_conflict")
        except SafeIoError:
            raise StagingError("transfer_member_write_failed") from None
    return verify_transfer(base, manifest)


def verify_transfer(root: Path | str, manifest: TransferManifest) -> TransferReceipt:
    """Re-read every staged member and reject partial or changed transfers."""

    if not isinstance(manifest, TransferManifest):
        raise StagingError("transfer_manifest_invalid")
    verified: list[TransferMember] = []
    for member in manifest.members:
        try:
            payload = read_stable_bytes(
                _member_path(Path(root), member.logical_location)
            )
        except SafeIoError:
            raise StagingError("transfer_member_unavailable") from None
        _verify_payload(member, payload)
        verified.append(member)
    return TransferReceipt(manifest.direction, manifest.identity, tuple(verified), True)


def read_verified_transfer(
    root: Path | str, manifest: TransferManifest
) -> dict[PortableLocation, bytes]:
    """Return bytes only after a complete identity-bound verification pass."""

    verify_transfer(root, manifest)
    result: dict[PortableLocation, bytes] = {}
    for member in manifest.members:
        try:
            result[member.logical_location] = read_stable_bytes(
                _member_path(Path(root), member.logical_location)
            )
        except SafeIoError:
            raise StagingError("transfer_changed_after_verification") from None
        _verify_payload(member, result[member.logical_location])
    return result


def _member_path(root: Path, location: PortableLocation) -> Path:
    try:
        candidate = root.joinpath(*location.logical_path.split("/"))
        ensure_safe_directory(candidate.parent)
        return candidate
    except (OSError, UnsafeFilesystemPath, PortablePathError):
        raise StagingError("transfer_location_unavailable") from None


def _verify_payload(member: TransferMember, payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise StagingError("transfer_payload_invalid")
    observed = ContentIdentity("sha256", hashlib.sha256(payload).hexdigest())
    if len(payload) != member.byte_count or observed != member.content_identity:
        raise StagingError("transfer_member_identity_mismatch")
