"""Immutable experiment manifests, exact diffs, and derivation evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    freeze_json_value,
    identity_fields,
    record_reference,
    require_identifier,
    require_text,
    thaw_json,
)
from temper_ml.store.canonical_json import dumps_canonical_json

EXPERIMENT_MANIFEST_PROJECTION = HashProjection("experiment.manifest", "v1")


class DiffOperation(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


class ReproductionMode(str, Enum):
    SCIENTIFIC_DERIVATION = "scientific_derivation"
    ADAPTED_REPRODUCTION = "adapted_reproduction"


class _Missing:
    pass


MISSING = _Missing()


def _require_reference(
    field: str, value: RecordReference, record_type: str
) -> RecordReference:
    if not isinstance(value, RecordReference) or value.record_type != record_type:
        raise RecordValidationError(f"{field} must reference {record_type}")
    return value


@dataclass(frozen=True)
class ManifestChange:
    """One replayable RFC 6901-addressed manifest operation."""

    path: str
    operation: DiffOperation
    before: Any = MISSING
    after: Any = MISSING

    def __post_init__(self) -> None:
        _parse_pointer(self.path)
        if not isinstance(self.operation, DiffOperation):
            raise RecordValidationError("manifest change operation is invalid")
        if self.path == "" and self.operation in (
            DiffOperation.ADD,
            DiffOperation.REMOVE,
        ):
            raise RecordValidationError("the object manifest root can only be replaced")
        if self.operation is DiffOperation.ADD:
            if self.before is not MISSING or self.after is MISSING:
                raise RecordValidationError("add requires only an after value")
        elif self.operation is DiffOperation.REMOVE:
            if self.before is MISSING or self.after is not MISSING:
                raise RecordValidationError("remove requires only a before value")
        elif self.before is MISSING or self.after is MISSING:
            raise RecordValidationError("replace requires before and after values")
        if self.before is not MISSING:
            object.__setattr__(
                self,
                "before",
                freeze_json_value(self.before, field="manifest change before"),
            )
        if self.after is not MISSING:
            object.__setattr__(
                self,
                "after",
                freeze_json_value(self.after, field="manifest change after"),
            )
        if self.operation is DiffOperation.REPLACE and _canonical_equal(
            self.before, self.after
        ):
            raise RecordValidationError("replace change must alter the value")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "path": self.path,
            "operation": self.operation.value,
        }
        if self.before is not MISSING:
            value["before"] = thaw_json(self.before)
        if self.after is not MISSING:
            value["after"] = thaw_json(self.after)
        return value


@dataclass(frozen=True)
class ManifestDiff(TypedRecord):
    """Exact, deterministic changes between two immutable manifests."""

    RECORD_TYPE: ClassVar[str] = "manifest_diff"

    diff_id: str
    parent_manifest_identity: ContentIdentity
    derived_manifest_identity: ContentIdentity
    changes: tuple[ManifestChange, ...]

    def __post_init__(self) -> None:
        require_identifier("diff_id", self.diff_id)
        if not isinstance(
            self.parent_manifest_identity, ContentIdentity
        ) or not isinstance(self.derived_manifest_identity, ContentIdentity):
            raise RecordValidationError(
                "manifest identities must be content identities"
            )
        if self.parent_manifest_identity == self.derived_manifest_identity:
            raise RecordValidationError(
                "manifest diff must change the manifest identity"
            )
        if not isinstance(self.changes, tuple) or not self.changes:
            raise RecordValidationError("manifest diff must contain changes")
        if any(not isinstance(change, ManifestChange) for change in self.changes):
            raise RecordValidationError("manifest diff contains an invalid change")
        ordered = tuple(sorted(self.changes, key=lambda change: change.path))
        paths = tuple(change.path for change in ordered)
        if len(set(paths)) != len(paths):
            raise RecordValidationError("manifest diff paths must be unique")
        for index, path in enumerate(paths):
            for other in paths[index + 1 :]:
                if _is_pointer_ancestor(path, other):
                    raise RecordValidationError("manifest diff paths must not overlap")
        object.__setattr__(self, "changes", ordered)

    @classmethod
    def between(
        cls,
        diff_id: str,
        parent: Mapping[str, Any],
        derived: Mapping[str, Any],
    ) -> "ManifestDiff":
        parent_value = thaw_json(freeze_json_value(parent, field="parent manifest"))
        derived_value = thaw_json(freeze_json_value(derived, field="derived manifest"))
        if not isinstance(parent_value, dict) or not isinstance(derived_value, dict):
            raise RecordValidationError("experiment manifests must be JSON objects")
        changes: list[ManifestChange] = []
        _diff_values("", parent_value, derived_value, changes)
        if not changes:
            raise RecordValidationError("cannot create a no-op manifest diff")
        return cls(
            diff_id=diff_id,
            parent_manifest_identity=manifest_identity(parent_value),
            derived_manifest_identity=manifest_identity(derived_value),
            changes=tuple(changes),
        )

    def apply(self, parent: Mapping[str, Any]) -> dict[str, Any]:
        value = thaw_json(freeze_json_value(parent, field="parent manifest"))
        if not isinstance(value, dict):
            raise RecordValidationError("parent manifest must be a JSON object")
        if manifest_identity(value) != self.parent_manifest_identity:
            raise RecordValidationError("parent manifest identity mismatch")
        result: Any = value
        for change in self.changes:
            result = _apply_change(result, change)
        if not isinstance(result, dict):
            raise RecordValidationError("derived manifest must be a JSON object")
        if manifest_identity(result) != self.derived_manifest_identity:
            raise RecordValidationError("derived manifest identity mismatch")
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "diff_id": self.diff_id,
            "parent_manifest_identity": identity_fields(self.parent_manifest_identity),
            "derived_manifest_identity": identity_fields(
                self.derived_manifest_identity
            ),
            "changes": [change.to_dict() for change in self.changes],
        }


@dataclass(frozen=True)
class Experiment(TypedRecord):
    """Immutable scientific intention; execution attempts are separate Runs."""

    RECORD_TYPE: ClassVar[str] = "experiment"

    experiment_id: str
    project: RecordReference
    project_policy: RecordReference
    task_definition: RecordReference
    dataset_version: ContentIdentity
    base_model_revision: RecordReference
    tokenizer_identity: ContentIdentity
    recipe: RecordReference
    recipe_resolution: RecordReference
    evaluation_policy: ContentIdentity
    compatibility_group: RecordReference
    hardware_requirements: RecordReference
    execution_target: RecordReference

    def __post_init__(self) -> None:
        require_identifier("experiment_id", self.experiment_id)
        for field, record_type in (
            ("project", "project"),
            ("project_policy", "project_policy"),
            ("task_definition", "task_definition"),
            ("base_model_revision", "base_model_revision"),
            ("recipe", "recipe"),
            ("recipe_resolution", "recipe_resolution"),
            ("compatibility_group", "compatibility_group"),
            ("hardware_requirements", "hardware_requirements"),
            ("execution_target", "execution_target"),
        ):
            _require_reference(field, getattr(self, field), record_type)
        for field in (
            "dataset_version",
            "tokenizer_identity",
            "evaluation_policy",
        ):
            if not isinstance(getattr(self, field), ContentIdentity):
                raise RecordValidationError(f"{field} must be a content identity")

    def scientific_manifest(self) -> dict[str, object]:
        payload = self.to_payload()
        del payload["experiment_id"]
        return payload

    @property
    def manifest_identity(self) -> ContentIdentity:
        return manifest_identity(self.scientific_manifest())

    def to_payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "project": self.project.to_dict(),
            "project_policy": self.project_policy.to_dict(),
            "task_definition": self.task_definition.to_dict(),
            "dataset_version": identity_fields(self.dataset_version),
            "base_model_revision": self.base_model_revision.to_dict(),
            "tokenizer_identity": identity_fields(self.tokenizer_identity),
            "recipe": self.recipe.to_dict(),
            "recipe_resolution": self.recipe_resolution.to_dict(),
            "evaluation_policy": identity_fields(self.evaluation_policy),
            "compatibility_group": self.compatibility_group.to_dict(),
            "hardware_requirements": self.hardware_requirements.to_dict(),
            "execution_target": self.execution_target.to_dict(),
        }


@dataclass(frozen=True)
class ExperimentDerivation(TypedRecord):
    """Immutable parent/child lineage with a verified exact manifest diff."""

    RECORD_TYPE: ClassVar[str] = "experiment_derivation"

    derivation_id: str
    parent_experiment: Experiment
    derived_experiment: Experiment
    reproduction_mode: ReproductionMode
    reason_code: str
    reason: str
    manifest_diff: ManifestDiff

    def __post_init__(self) -> None:
        require_identifier("derivation_id", self.derivation_id)
        if not isinstance(self.parent_experiment, Experiment) or not isinstance(
            self.derived_experiment, Experiment
        ):
            raise RecordValidationError(
                "experiment derivation must embed parent and derived Experiments"
            )
        if (
            self.parent_experiment.experiment_id
            == self.derived_experiment.experiment_id
            or self.parent_experiment.identity == self.derived_experiment.identity
        ):
            raise RecordValidationError("derived experiment must differ from parent")
        if not isinstance(self.reproduction_mode, ReproductionMode):
            raise RecordValidationError("reproduction_mode is invalid")
        require_identifier("reason_code", self.reason_code)
        require_text("reason", self.reason)
        if not isinstance(self.manifest_diff, ManifestDiff):
            raise RecordValidationError("manifest_diff must be a ManifestDiff")
        if (
            self.manifest_diff.parent_manifest_identity
            != self.parent_experiment.manifest_identity
            or self.manifest_diff.derived_manifest_identity
            != self.derived_experiment.manifest_identity
        ):
            raise RecordValidationError(
                "manifest diff identities do not match the embedded experiments"
            )
        if (
            self.manifest_diff.apply(self.parent_experiment.scientific_manifest())
            != self.derived_experiment.scientific_manifest()
        ):
            raise RecordValidationError(
                "manifest diff does not reproduce the embedded derived experiment"
            )

    @property
    def parent_reference(self) -> RecordReference:
        return record_reference(
            self.parent_experiment, self.parent_experiment.experiment_id
        )

    @property
    def derived_reference(self) -> RecordReference:
        return record_reference(
            self.derived_experiment, self.derived_experiment.experiment_id
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "derivation_id": self.derivation_id,
            "parent_experiment": self.parent_experiment.to_dict(),
            "derived_experiment": self.derived_experiment.to_dict(),
            "reproduction_mode": self.reproduction_mode.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "manifest_diff": self.manifest_diff.to_dict(),
        }


def derive_experiment(
    parent: Experiment,
    derived: Experiment,
    *,
    derivation_id: str,
    diff_id: str,
    reason_code: str,
    reason: str,
    reproduction_mode: ReproductionMode = ReproductionMode.SCIENTIFIC_DERIVATION,
) -> ExperimentDerivation:
    """Create checked lineage evidence for a materially changed experiment."""

    if not isinstance(parent, Experiment) or not isinstance(derived, Experiment):
        raise RecordValidationError("experiment derivation requires Experiment records")
    if parent.experiment_id == derived.experiment_id:
        raise RecordValidationError("derived experiment requires a new logical id")
    diff = ManifestDiff.between(
        diff_id, parent.scientific_manifest(), derived.scientific_manifest()
    )
    if diff.apply(parent.scientific_manifest()) != derived.scientific_manifest():
        raise RecordValidationError(
            "manifest diff does not reproduce derived experiment"
        )
    return ExperimentDerivation(
        derivation_id=derivation_id,
        parent_experiment=parent,
        derived_experiment=derived,
        reproduction_mode=reproduction_mode,
        reason_code=reason_code,
        reason=reason,
        manifest_diff=diff,
    )


def manifest_identity(manifest: Mapping[str, Any]) -> ContentIdentity:
    """Hash one canonical scientific manifest independent of its location."""

    value = thaw_json(freeze_json_value(manifest, field="manifest"))
    if not isinstance(value, dict):
        raise RecordValidationError("manifest must be a JSON object")
    return content_identity(EXPERIMENT_MANIFEST_PROJECTION, value)


def _diff_values(
    path: str, parent: Any, derived: Any, changes: list[ManifestChange]
) -> None:
    if isinstance(parent, dict) and isinstance(derived, dict):
        for key in sorted(set(parent) | set(derived)):
            child_path = f"{path}/{_escape_pointer(key)}"
            if key not in parent:
                changes.append(
                    ManifestChange(child_path, DiffOperation.ADD, after=derived[key])
                )
            elif key not in derived:
                changes.append(
                    ManifestChange(child_path, DiffOperation.REMOVE, before=parent[key])
                )
            else:
                _diff_values(child_path, parent[key], derived[key], changes)
        return
    if not _canonical_equal(parent, derived):
        changes.append(
            ManifestChange(path, DiffOperation.REPLACE, before=parent, after=derived)
        )


def _apply_change(root: Any, change: ManifestChange) -> Any:
    tokens = _parse_pointer(change.path)
    if not tokens:
        if change.operation is DiffOperation.ADD:
            raise RecordValidationError("cannot add an already present root")
        if not _canonical_equal(root, change.before):
            raise RecordValidationError("manifest diff before value mismatch")
        if change.operation is DiffOperation.REMOVE:
            raise RecordValidationError("cannot remove the manifest root")
        return thaw_json(change.after)
    if not isinstance(root, dict):
        raise RecordValidationError("manifest diff traverses a non-object")
    parent = root
    for token in tokens[:-1]:
        child = parent.get(token, MISSING)
        if not isinstance(child, dict):
            raise RecordValidationError("manifest diff path does not exist")
        parent = child
    key = tokens[-1]
    present = key in parent
    if change.operation is DiffOperation.ADD:
        if present:
            raise RecordValidationError("manifest add path already exists")
        parent[key] = thaw_json(change.after)
    else:
        if not present or not _canonical_equal(parent[key], change.before):
            raise RecordValidationError("manifest diff before value mismatch")
        if change.operation is DiffOperation.REMOVE:
            del parent[key]
        else:
            parent[key] = thaw_json(change.after)
    return root


def _canonical_equal(left: Any, right: Any) -> bool:
    if left is MISSING or right is MISSING:
        return left is right
    return dumps_canonical_json(thaw_json(left)) == dumps_canonical_json(
        thaw_json(right)
    )


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _parse_pointer(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise RecordValidationError("manifest diff path must be a string")
    if value == "":
        return ()
    if not value.startswith("/"):
        raise RecordValidationError("manifest diff path must be an RFC 6901 pointer")
    tokens: list[str] = []
    for raw in value[1:].split("/"):
        index = 0
        decoded = ""
        while index < len(raw):
            if raw[index] != "~":
                decoded += raw[index]
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in "01":
                raise RecordValidationError("manifest diff path has invalid escaping")
            decoded += "~" if raw[index + 1] == "0" else "/"
            index += 2
        tokens.append(decoded)
    return tuple(tokens)


def _is_pointer_ancestor(parent: str, child: str) -> bool:
    parent_tokens = _parse_pointer(parent)
    child_tokens = _parse_pointer(child)
    return (
        len(parent_tokens) < len(child_tokens)
        and child_tokens[: len(parent_tokens)] == parent_tokens
    )
