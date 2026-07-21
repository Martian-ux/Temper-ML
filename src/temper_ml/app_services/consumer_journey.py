"""Consumer-facing fixture demo and real local adapter journey orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event, RLock, Thread
import time
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from temper_ml.app_services.datasets import (
    CsvDatasetAdapter,
    DatasetAnalysis,
    DatasetImportRequest,
    DatasetPreflightError,
    DatasetService,
    HuggingFaceRowsDatasetAdapter,
    ImportedSource,
    JsonDatasetAdapter,
    JsonlDatasetAdapter,
    PreparedDataset,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.experiments import (
    ExperimentFreezeRequest,
    ExperimentService,
)
from temper_ml.app_services.fixture_journey import FixtureJourneyService
from temper_ml.app_services.local_use import (
    AdapterExportRequest,
    LocalUseRequest,
    LocalUseService,
)
from temper_ml.domain.local_use import AdapterExport, LocalUseSession
from temper_ml.app_services.projects import (
    OpenedProject,
    ProjectCreateRequest,
    ProjectService,
)
from temper_ml.app_services.runs import (
    RunExecutionResult,
    RunLaunchRequest,
    RunLifecycleStatus,
    RunRecoveryRequest,
    RunService,
)
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import CompatibilityGroup, RuntimeTargetConstraint
from temper_ml.domain.datasets import (
    DeduplicationRule,
    FieldMapping,
    FilterRule,
    RendererKind,
    RendererSpec,
    SplitPart,
    SplitRule,
    renderer_identity,
)
from temper_ml.domain.hardware import ExecutionTarget, HardwareRequirements
from temper_ml.domain.policies import BaselinePolicy, PerModelBaseline
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import record_reference
from temper_ml.domain.runs import EvaluationMode
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.runtime.fixture_inference import InferenceSettings
from temper_ml.runtime.library_adapter import (
    LibraryAdapter,
    LibraryInferenceRuntime,
    LibraryRuntimeSources,
)
from temper_ml.runtime.library_backend import (
    LibraryBackend,
    LibraryCapability,
    LibraryRuntimeError,
    TransformersPeftBackend,
)
from temper_ml.runtime.paths import WindowsWslPathMap
from temper_ml.runtime.preflight import (
    EstimateComponents,
    PreflightEstimate,
    estimate_resources,
    preflight,
)
from temper_ml.runtime.recipe_resolution import RecipeCatalogEntry, RecipeResolver
from temper_ml.runtime.worker_port import WslWorkerLaunchSpec
from temper_ml.runtime.wsl_backend import WslWorkerBackend, WslWorkerConfig
from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore
from temper_ml.store.safe_io import SafeIoError, read_stable_bytes, replace_bytes


MAX_HUGGING_FACE_BYTES = 256 * 1024 * 1024
MAX_HUGGING_FACE_ROWS = 10_000
DEFAULT_HUGGING_FACE_ROWS = 64
CONSUMER_SESSION_SCHEMA = "v1"
CONSUMER_SESSION_RELATIVE_PATH = Path(".temper/derived/private-consumer/session.json")


class ConsumerJourneyError(ApplicationServiceError):
    """A stable error plus bounded public-safe recovery details."""

    def __init__(self, code: str, details: Mapping[str, object] | None = None) -> None:
        self.details = dict(details or {})
        super().__init__(code)


class TokenizerFactory(Protocol):
    def __call__(self, source: Path, identity: ContentIdentity) -> "LocalTokenizer": ...


class LocalTokenizer:
    """Local-only Transformers tokenizer implementing DatasetService's port."""

    def __init__(self, source: Path, identity: ContentIdentity) -> None:
        try:
            import transformers  # type: ignore[import-not-found]

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(source), local_files_only=True, trust_remote_code=False
            )
        except Exception:
            raise ConsumerJourneyError(
                "tokenizer_preflight_failed",
                {
                    "action": (
                        "Choose a complete local tokenizer directory and install "
                        "the runtime extra."
                    ),
                    "source_kind": "local_directory",
                },
            ) from None
        self._tokenizer = tokenizer
        self._identity = identity

    @property
    def identity(self) -> ContentIdentity:
        return self._identity

    def count_tokens(self, text: str) -> int:
        try:
            return len(self._tokenizer.encode(text, add_special_tokens=True))
        except Exception:
            raise ConsumerJourneyError("tokenizer_count_failed") from None


class HuggingFaceDatasetClient:
    """Bounded public Hugging Face URL/config/split and repository-file importer."""

    def fetch(
        self,
        *,
        dataset_url: str,
        config: str,
        split: str,
        file_path: str | None,
        row_limit: int,
        source_mode: str = "config_split",
    ) -> tuple[ImportedSource, dict[str, object]]:
        dataset_id, url_file = _parse_hugging_face_dataset_url(dataset_url)
        if row_limit < 1 or row_limit > MAX_HUGGING_FACE_ROWS:
            raise ConsumerJourneyError("hugging_face_row_limit_invalid")
        if source_mode not in {"config_split", "repository_file"}:
            raise ConsumerJourneyError(
                "hugging_face_source_mode_invalid",
                {
                    "action": (
                        "Choose either config and split rows or one repository file."
                    )
                },
            )
        selected_file = file_path.strip() if isinstance(file_path, str) else ""
        if selected_file and url_file and selected_file != url_file:
            raise ConsumerJourneyError(
                "hugging_face_file_ambiguous",
                {"action": "Use one repository file path or one file URL, not both."},
            )
        selected_file = selected_file or url_file or ""
        if source_mode == "repository_file":
            if not selected_file:
                raise ConsumerJourneyError(
                    "hugging_face_file_required",
                    {"action": "Enter a repository-relative JSON, JSONL, or CSV file."},
                )
            if config or split:
                raise ConsumerJourneyError(
                    "hugging_face_source_fields_conflict",
                    {
                        "action": (
                            "Clear config and split when repository-file mode is "
                            "selected."
                        )
                    },
                )
            if (
                selected_file.startswith(("/", "\\"))
                or ".." in Path(selected_file).parts
            ):
                raise ConsumerJourneyError("hugging_face_file_invalid")
            encoded = "/".join(
                quote(part, safe="") for part in selected_file.split("/")
            )
            url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{encoded}"
            data = _read_public_url(url, MAX_HUGGING_FACE_BYTES)
            adapter = _adapter_for_format(Path(selected_file).suffix.removeprefix("."))
            imported = adapter.load(data)
            imported, limited_from = _limit_imported_source(imported, row_limit)
            return imported, {
                "kind": "hugging_face_file",
                "dataset_id": dataset_id,
                "file": selected_file,
                "bytes": len(data),
                "available_rows": limited_from,
                "imported_rows": len(imported.rows),
                "row_limit": row_limit,
            }
        if selected_file:
            raise ConsumerJourneyError(
                "hugging_face_source_fields_conflict",
                {
                    "action": (
                        "Clear the repository file or select repository-file mode."
                    )
                },
            )
        if not config or not split:
            raise ConsumerJourneyError(
                "hugging_face_config_split_required",
                {
                    "action": (
                        "Provide both dataset config and split, or a repository file."
                    )
                },
            )
        rows: list[Mapping[str, object]] = []
        total = None
        while len(rows) < row_limit:
            length = min(100, row_limit - len(rows))
            query = (
                "https://datasets-server.huggingface.co/rows?dataset="
                f"{quote(dataset_id, safe='')}&config={quote(config, safe='')}"
                f"&split={quote(split, safe='')}&offset={len(rows)}&length={length}"
            )
            try:
                value = json.loads(_read_public_url(query, 16 * 1024 * 1024))
            except (UnicodeError, json.JSONDecodeError):
                raise ConsumerJourneyError(
                    "hugging_face_rows_invalid",
                    {"action": "Verify the public config and split, then retry."},
                ) from None
            if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
                raise ConsumerJourneyError("hugging_face_rows_invalid")
            page = value["rows"]
            total_value = value.get("num_rows_total")
            if isinstance(total_value, int) and total_value >= 0:
                total = total_value
            for entry in page:
                if len(rows) >= row_limit:
                    break
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("row"), dict
                ):
                    raise ConsumerJourneyError("hugging_face_rows_invalid")
                rows.append(entry["row"])
            if not page or (total is not None and len(rows) >= total):
                break
        imported = HuggingFaceRowsDatasetAdapter().load(rows)
        return imported, {
            "kind": "hugging_face_config_split",
            "dataset_id": dataset_id,
            "config": config,
            "split": split,
            "imported_rows": len(rows),
            "available_rows": total,
            "row_limit": row_limit,
        }


@dataclass(frozen=True)
class RealSetup:
    model_source: Path
    tokenizer_source: Path
    display_name: str
    model_family: str
    architecture: str
    revision: str
    license: str
    target: str


@dataclass
class OperationState:
    status: str = "idle"
    phase: str = "not_started"
    started_at: float | None = None
    finished_at: float | None = None
    step: int = 0
    total_steps: int = 0
    cancellation_requested: bool = False
    failure_code: str | None = None
    recovery_action: str | None = None

    def to_view(self) -> dict[str, object]:
        now = self.finished_at or time.monotonic()
        elapsed = 0 if self.started_at is None else max(0, int(now - self.started_at))
        return {
            "status": self.status,
            "phase": self.phase,
            "elapsed_seconds": elapsed,
            "step": self.step,
            "total_steps": self.total_steps,
            "cancellation_requested": self.cancellation_requested,
            "failure_code": self.failure_code,
            "recovery_action": self.recovery_action,
        }


@dataclass
class RealState:
    setup: RealSetup | None = None
    opened: OpenedProject | None = None
    model: BaseModelRevision | None = None
    prepared: PreparedDataset | None = None
    dataset_view: dict[str, object] | None = None
    backend: LibraryBackend | None = None
    adapter: LibraryAdapter | None = None
    sources: LibraryRuntimeSources | None = None
    target: ExecutionTarget | None = None
    requirements: HardwareRequirements | None = None
    capability: LibraryCapability | None = None
    resolution: RecipeResolution | None = None
    recipe: Recipe | None = None
    group: CompatibilityGroup | None = None
    estimate: PreflightEstimate | None = None
    launch: RunLaunchRequest | None = None
    launch_consumed: bool = False
    result: RunExecutionResult | None = None
    local_result: dict[str, object] | None = None
    selected: bool = False
    recovery_required: bool = False
    operation: OperationState = field(default_factory=OperationState)
    store_snapshot: (
        tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]
        | None
    ) = None


class ConsumerJourneyService:
    """One local UI session with explicit fixture-demo or real-library mode."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        backend_factory: Callable[[RealSetup], LibraryBackend] | None = None,
        tokenizer_factory: TokenizerFactory | None = None,
        hugging_face_client: HuggingFaceDatasetClient | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.demo = FixtureJourneyService(project_root)
        self.real = RealState()
        self.mode: str | None = None
        self._backend_factory = backend_factory or _default_backend
        self._tokenizer_factory = tokenizer_factory or LocalTokenizer
        self._hugging_face = hugging_face_client or HuggingFaceDatasetClient()
        self._lock = RLock()
        self._cancel = Event()
        self._thread: Thread | None = None
        self._restore_private_session()

    def setup_project(
        self,
        *,
        mode: str = "fixture_demo",
        model_source: str | None = None,
        tokenizer_source: str | None = None,
        display_name: str = "Local model",
        model_family: str = "local-causal-lm",
        architecture: str = "causal-lm",
        revision: str = "local-revision",
        license_name: str = "user-confirmed-local-license",
        target: str = "native_local",
    ) -> dict[str, object]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ConsumerJourneyError(
                    "mode_change_blocked_by_active_run",
                    {"action": "Cancel or finish the active real run first."},
                )
        if mode == "fixture_demo":
            result = self.demo.setup_project()
            with self._lock:
                self.mode = mode
                self._persist_private_session()
            return {**result, "mode": mode, "demo": True}
        if mode != "real_local":
            raise ConsumerJourneyError("training_mode_invalid")
        if not model_source or not tokenizer_source:
            raise ConsumerJourneyError(
                "local_model_sources_required",
                {"action": "Choose local model and tokenizer directories."},
            )
        model_path = _local_directory(model_source, "model_source_invalid")
        tokenizer_path = _local_directory(tokenizer_source, "tokenizer_source_invalid")
        if target not in {"native_local", "wsl_rocm"}:
            raise ConsumerJourneyError("execution_target_invalid")
        with self._lock:
            self.real = RealState(
                setup=RealSetup(
                    model_path,
                    tokenizer_path,
                    display_name,
                    model_family,
                    architecture,
                    revision,
                    license_name,
                    target,
                )
            )
            self.mode = mode
            self._persist_private_session()
        return {
            "status": "configured",
            "mode": mode,
            "demo": False,
            "target": target,
            "model_source": "local_directory",
            "tokenizer_source": "local_directory",
        }

    def import_dataset(
        self,
        *,
        source_format: str = "fixture",
        source_text: str | None = None,
        options: Mapping[str, object] | None = None,
        source_bytes: bytes | None = None,
    ) -> dict[str, object]:
        if self.mode == "fixture_demo":
            if source_bytes is not None:
                try:
                    source_text = source_bytes.decode("utf-8")
                except UnicodeError:
                    raise ConsumerJourneyError("dataset_source_not_utf8") from None
            return self.demo.import_dataset(
                source_format=source_format, source_text=source_text
            )
        if self.mode != "real_local":
            raise ConsumerJourneyError("consumer_setup_required")
        self._ensure_no_active_run(
            "dataset_import_blocked_by_active_run",
            "Cancel or finish the active real run before replacing its dataset.",
        )
        setup = self._require_real_setup()
        inputs = dict(options or {})
        weights_identity, model_bytes = _directory_identity(setup.model_source)
        tokenizer_identity, _ = _directory_identity(setup.tokenizer_source)
        tokenizer = self._tokenizer_factory(setup.tokenizer_source, tokenizer_identity)
        mapping = FieldMapping(
            _text_option(inputs, "context_field", "context"),
            _text_option(inputs, "completion_field", "completion"),
            _optional_field(inputs.get("supporting_context_field")),
            _optional_field(inputs.get("cot_field")),
            _optional_field(inputs.get("output_field")),
        )
        renderer_name = _text_option(inputs, "renderer", "trace_completion")
        try:
            renderer = RendererSpec(RendererKind(renderer_name))
        except ValueError:
            raise ConsumerJourneyError("renderer_invalid") from None
        imported, provenance = self._load_source(
            source_format, source_text, source_bytes, inputs
        )
        attempt = _new_attempt_id("import")
        request = DatasetImportRequest(
            version_id=f"dataset-real-consumer-{attempt}",
            field_mapping=mapping,
            renderer=renderer,
            filter_rule=FilterRule(
                _int_option(inputs, "minimum_characters", 1, minimum=0),
                _optional_positive_int(inputs.get("maximum_characters")),
                _optional_positive_int(inputs.get("maximum_tokens", 4096)),
            ),
            deduplication_rule=DeduplicationRule(),
            split_rule=SplitRule(
                _int_option(inputs, "split_seed", 17, minimum=0),
                (
                    SplitPart(
                        "train",
                        _int_option(inputs, "train_weight", 9, minimum=1),
                    ),
                    SplitPart(
                        "validation",
                        _int_option(inputs, "validation_weight", 1, minimum=1),
                    ),
                ),
            ),
            tokenizer=tokenizer,
            preview_limit=min(_int_option(inputs, "preview_limit", 3, minimum=0), 10),
            required_non_empty_splits=("train",),
        )
        service = DatasetService(self.project_root)
        analysis = service.analyze_source(imported, request)
        _require_trainable_analysis(analysis)
        rendering_contract = renderer_identity(mapping, renderer)
        opened, model = _create_real_project(
            self.project_root,
            setup,
            rendering_contract,
            weights_identity,
            tokenizer_identity,
            attempt,
        )
        prepared = service.import_source(imported, request)
        with self._lock:
            self.real.opened = opened
            self.real.model = model
            self.real.prepared = prepared
            self.real.dataset_view = {
                "dataset_version": record_reference(prepared.version).to_dict(),
                "statistics": prepared.version.statistics.to_dict(),
                "analysis": analysis.to_view(),
                "previews": [
                    _bounded_preview(item.to_dict()) for item in prepared.previews
                ],
                "private_preview": True,
                "source": provenance,
                "tokenizer": "real_local_tokenizer",
                "model_bytes": model_bytes,
                "rendered_bytes_count": prepared.version.rendered_bytes_count,
            }
            self.real.backend = None
            self.real.adapter = None
            self.real.sources = None
            self.real.target = None
            self.real.requirements = None
            self.real.capability = None
            self.real.resolution = None
            self.real.recipe = None
            self.real.group = None
            self.real.estimate = None
            self.real.launch = None
            self.real.launch_consumed = False
            self.real.result = None
            self.real.local_result = None
            self.real.selected = False
            self.real.recovery_required = False
            self.real.operation = OperationState()
            self._persist_private_session()
        return dict(self.real.dataset_view)

    def _load_source(
        self,
        source_format: str,
        source_text: str | None,
        source_bytes: bytes | None,
        options: Mapping[str, object],
    ) -> tuple[ImportedSource, dict[str, object]]:
        if source_format == "hugging_face":
            return self._hugging_face.fetch(
                dataset_url=_text_option(options, "dataset_url", ""),
                config=_text_option(options, "config", ""),
                split=_text_option(options, "split", ""),
                file_path=_optional_text(options.get("file_path")),
                row_limit=_int_option(
                    options,
                    "row_limit",
                    DEFAULT_HUGGING_FACE_ROWS,
                    minimum=1,
                ),
                source_mode=_text_option(
                    options, "hugging_face_source_mode", "config_split"
                ),
            )
        data = source_bytes
        if data is None and source_text is not None:
            data = source_text.encode("utf-8")
        if not data:
            raise ConsumerJourneyError("dataset_source_required")
        adapter = _adapter_for_format(source_format)
        imported = adapter.load(data)
        row_limit = _int_option(
            options, "row_limit", DEFAULT_HUGGING_FACE_ROWS, minimum=1
        )
        imported, limited_from = _limit_imported_source(imported, row_limit)
        return imported, {
            "kind": "local_file" if source_bytes is not None else "pasted_text",
            "format": source_format,
            "bytes": len(data),
            "available_rows": limited_from,
            "imported_rows": len(imported.rows),
            "row_limit": row_limit,
        }

    def resolve_candidates(
        self, options: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        if self.mode == "fixture_demo":
            return self.demo.resolve_candidates()
        if self.mode != "real_local":
            raise ConsumerJourneyError("consumer_setup_required")
        self._ensure_no_active_run(
            "candidate_resolution_blocked_by_active_run",
            "Cancel or finish the active real run before resolving another attempt.",
        )
        setup, opened, model, prepared = self._real_dataset_context()
        inputs = dict(options or {})
        target_modules = tuple(
            sorted(
                set(
                    item.strip()
                    for item in _text_option(inputs, "target_modules", "c_attn").split(
                        ","
                    )
                    if item.strip()
                )
            )
        )
        if not target_modules:
            raise ConsumerJourneyError(
                "target_modules_required",
                {"action": "Enter at least one LoRA target module name."},
            )
        try:
            backend = self._backend_factory(setup)
            capability = backend.probe()
            source_preflight = getattr(backend, "preflight_sources", None)
            source_facts = (
                dict(source_preflight(setup.model_source, setup.tokenizer_source))
                if callable(source_preflight)
                else {"status": "backend_did_not_expose_source_probe"}
            )
            module_preflight = getattr(backend, "preflight_target_modules", None)
            target_module_facts = (
                dict(module_preflight(setup.model_source, target_modules))
                if callable(module_preflight)
                else {"status": "backend_did_not_expose_target_module_probe"}
            )
        except ConsumerJourneyError:
            raise
        except LibraryRuntimeError as exc:
            raise ConsumerJourneyError(
                exc.code,
                {
                    "action": _library_recovery_action(exc.code, setup.target),
                    "fallback_used": False,
                },
            ) from None
        except Exception:
            raise ConsumerJourneyError(
                "library_preflight_failed",
                {"action": "Install the runtime extra and verify the selected target."},
            ) from None
        target_class = (
            "wsl_rocm"
            if setup.target == "wsl_rocm"
            else f"native_{capability.accelerator_backend}"
        )
        attempt = _new_attempt_id("resolve")
        target = ExecutionTarget(
            f"target-real-consumer-{attempt}",
            target_class,
            "wsl" if setup.target == "wsl_rocm" else "windows",
            capability.accelerator_backend,
            _identity("real-runtime-contract-v1"),
            capability.capabilities,
            {"local_only": True, "network_required": False, "explicit_target": True},
        )
        precision = _text_option(inputs, "precision", "fp32")
        quantization = _text_option(inputs, "quantization", "none")
        requirements = HardwareRequirements(
            f"requirements-real-consumer-{attempt}",
            (target_class,),
            (capability.accelerator_backend,),
            0,
            1,
            (precision,),
            () if quantization == "none" else (quantization,),
            ("lora", "peft", "transformers"),
            {"local_only": True, "network_required": False},
        )
        recipe = Recipe(
            f"recipe-real-consumer-{attempt}",
            "local-lora",
            "v1",
            "bounded",
            "small",
            "conservative",
            quantization,
            "bounded-steps",
            "every-step",
            "none",
            "full",
            {},
        )
        entry = RecipeCatalogEntry(
            recipe,
            {
                "adapter_type": "lora",
                "target_modules": list(target_modules),
                "rank": _int_option(inputs, "rank", 4, minimum=1),
                "alpha": _int_option(inputs, "alpha", 8, minimum=1),
                "dropout": _decimal_option(
                    inputs,
                    "dropout",
                    "0.05",
                    minimum=Decimal("0"),
                    maximum_exclusive=Decimal("1"),
                ),
                "learning_rate": _decimal_option(
                    inputs,
                    "learning_rate",
                    "0.0002",
                    minimum=Decimal("0.000000000000000001"),
                ),
                "effective_batch_size": _int_option(
                    inputs, "effective_batch_size", 1, minimum=1
                ),
                "sequence_length": _int_option(
                    inputs, "sequence_length", 256, minimum=8
                ),
                "optimizer": "adamw",
                "precision": precision,
                "gradient_accumulation": _int_option(
                    inputs, "gradient_accumulation", 1, minimum=1
                ),
                "seed": _int_option(inputs, "seed", 17, minimum=0),
                "schedule": "linear",
                "training_steps": _int_option(
                    inputs, "training_steps", 2, minimum=1, maximum=100
                ),
                "checkpoint_cadence": _int_option(
                    inputs, "checkpoint_cadence", 1, minimum=1
                ),
                "quantization": quantization,
                "library_versions": dict(capability.library_versions),
            },
        )
        resolution = RecipeResolver().resolve(
            entry,
            base_model_revision=model,
            hardware_requirements=requirements,
            execution_target=target,
        )
        group = CompatibilityGroup(
            f"group-real-consumer-{attempt}",
            record_reference(model),
            model.tokenizer_identity,
            opened.task_definition.rendering_contract,
            "lora",
            resolution.target_modules,
            (
                RuntimeTargetConstraint(
                    target_class,
                    capability.accelerator_backend,
                    target.runtime_contract,
                    ("lora", "peft", "transformers"),
                ),
            ),
            (),
        )
        sources = LibraryRuntimeSources(
            setup.model_source,
            setup.tokenizer_source,
            (
                self.project_root / ".temper" / "derived" / "private-library-staging"
            ).resolve(),
            target_class,
            record_reference(model),
            model.tokenizer_identity,
        )
        adapter = LibraryAdapter(
            backend,
            sources,
            capability=capability,
            cancellation_requested=self._cancel.is_set,
            progress_callback=self._record_progress,
        )
        profile = adapter.capability_profile(f"profile-real-consumer-{attempt}", target)
        _, model_bytes = _directory_identity(setup.model_source)
        estimate = estimate_resources(
            resolution,
            EstimateComponents(
                base_model_bytes=model_bytes if capability.accelerator_count else 0,
                adapter_optimizer_bytes=(
                    max(1, resolution.rank * 1024 * 1024)
                    if capability.accelerator_count
                    else 0
                ),
                peak_activation_bytes=(
                    max(1, resolution.sequence_length * 1024 * 1024)
                    if capability.accelerator_count
                    else 0
                ),
                accelerator_runtime_overhead_bytes=0,
                dataset_bytes=len(prepared.rendered_bytes),
                host_runtime_overhead_bytes=(64 * 1024 * 1024 + model_bytes),
            ),
        )
        result = preflight(resolution, requirements, target, profile, estimate)
        if not result.ready:
            raise ConsumerJourneyError(
                "real_preflight_blocked",
                {
                    "preflight": result.to_view(),
                    "action": (
                        "Adjust the recipe or choose an explicitly supported target; "
                        "Temper will not silently fall back."
                    ),
                },
            )
        experiment = ExperimentService(self.project_root).freeze(
            ExperimentFreezeRequest(
                f"experiment-real-consumer-{attempt}",
                opened,
                prepared.version.identity,
                model,
                recipe,
                resolution,
                group,
                requirements,
                target,
            )
        )
        run_attempt = uuid4().hex[:16]
        launch = RunLaunchRequest(
            f"run-real-consumer-{run_attempt}",
            f"request-real-consumer-{run_attempt}",
            f"artifact-real-consumer-{run_attempt}",
            experiment,
            resolution,
            prepared,
            model,
            group,
            requirements,
            target,
            profile,
            estimate,
            EvaluationMode.NO_QUALITY_EVALUATION,
        )
        with self._lock:
            self.real.backend = backend
            self.real.adapter = adapter
            self.real.sources = sources
            self.real.target = target
            self.real.requirements = requirements
            self.real.capability = capability
            self.real.resolution = resolution
            self.real.recipe = recipe
            self.real.group = group
            self.real.estimate = estimate
            self.real.launch = launch
            self.real.launch_consumed = False
            self.real.result = None
            self.real.local_result = None
            self.real.selected = False
            self.real.recovery_required = False
            self.real.operation = OperationState()
            self._persist_private_session()
        return {
            "mode": "real_local",
            "demo": False,
            "execution_target": target.to_payload(),
            "capability": capability.to_public_facts(),
            "source_preflight": source_facts,
            "target_module_preflight": target_module_facts,
            "resolution": resolution.to_payload(),
            "preflight": result.to_view(),
            "fallback_used": False,
        }

    def launch_candidates(self) -> dict[str, object]:
        if self.mode == "fixture_demo":
            result = self.demo.launch_candidates()
            return {**result, "mode": "fixture_demo", "demo": True}
        if self.mode != "real_local":
            raise ConsumerJourneyError("consumer_setup_required")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ConsumerJourneyError("run_already_active")
            if self.real.launch is None or self.real.adapter is None:
                raise ConsumerJourneyError("real_preflight_required")
            recovery: RunRecoveryRequest | None = None
            if self.real.launch_consumed:
                prior = self.real.result
                resumable = (
                    tuple(
                        checkpoint
                        for checkpoint in prior.checkpoints
                        if checkpoint.resume_compatible
                    )
                    if prior is not None
                    and prior.status is RunLifecycleStatus.INTERRUPTED
                    else ()
                )
                if not resumable or prior is None:
                    raise ConsumerJourneyError(
                        "real_run_attempt_consumed",
                        {
                            "action": (
                                "Resolve the recipe again to create a new immutable "
                                "run attempt."
                            )
                        },
                    )
                suffix = uuid4().hex[:16]
                launch = replace(
                    self.real.launch,
                    run_id=f"run-real-recovery-{suffix}",
                    request_id=f"request-real-recovery-{suffix}",
                    artifact_id=f"artifact-real-recovery-{suffix}",
                )
                recovery = RunRecoveryRequest(
                    launch,
                    prior.run,
                    resumable[-1].checkpoint_identity,
                )
                self.real.launch = launch
            else:
                launch = self.real.launch
            adapter = self.real.adapter
            # Workspace polling runs concurrently with the library writer. Capture a
            # stable pre-launch view rather than enumerating atomic temp files while
            # the evidence store is being updated.
            self.real.store_snapshot = _real_store_summary(self.project_root)
            self.real.launch_consumed = True
            self._cancel.clear()
            self.real.operation = OperationState(
                status="running",
                phase="recovering" if recovery is not None else "training",
                started_at=time.monotonic(),
                total_steps=self.real.resolution.training_steps
                if self.real.resolution
                else 0,
            )
            self._thread = Thread(
                target=self._run_real,
                args=(launch, adapter, recovery),
                name="temper-real-training",
                daemon=True,
            )
            self._persist_private_session(best_effort=True)
            self._thread.start()
        return {
            "started": True,
            "recovery": recovery is not None,
            "mode": "real_local",
            "operation": self.real.operation.to_view(),
        }

    def cancel_run(self) -> dict[str, object]:
        if self.mode == "fixture_demo":
            raise ConsumerJourneyError("fixture_demo_cancellation_not_required")
        if self.mode != "real_local":
            raise ConsumerJourneyError("consumer_setup_required")
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                raise ConsumerJourneyError("run_not_active")
            self._cancel.set()
            self.real.operation.cancellation_requested = True
            self.real.operation.phase = "cancellation_requested"
            return {"accepted": True, "operation": self.real.operation.to_view()}

    def _run_real(
        self,
        launch: RunLaunchRequest,
        adapter: LibraryAdapter,
        recovery: RunRecoveryRequest | None,
    ) -> None:
        try:
            service = RunService(self.project_root, adapter=adapter)
            result = (
                service.recover(recovery)
                if recovery is not None
                else service.launch(launch)
            )
            with self._lock:
                self.real.result = result
                self.real.selected = result.status is RunLifecycleStatus.COMPLETED
                self.real.operation.status = result.status.value
                self.real.operation.phase = (
                    "artifact_verified"
                    if result.status is RunLifecycleStatus.COMPLETED
                    else result.status.value
                )
                self.real.operation.finished_at = time.monotonic()
                if result.status is RunLifecycleStatus.INTERRUPTED:
                    self.real.operation.recovery_action = (
                        "The latest compatible checkpoint is retained. Launch again "
                        "to create a new attempt that resumes from it."
                    )
                elif result.status is RunLifecycleStatus.CANCELLED:
                    self.real.operation.recovery_action = (
                        "The run was cancelled. Resolve again when you are ready "
                        "to create a new attempt."
                    )
                self._persist_private_session(best_effort=True)
        except Exception as exc:
            with self._lock:
                self.real.operation.status = "failed"
                self.real.operation.phase = "failed"
                self.real.operation.failure_code = getattr(
                    exc, "code", "real_training_failed"
                )
                self.real.operation.recovery_action = (
                    "Inspect the public-safe failure code, correct the prerequisite, "
                    "then resolve again to create a new run attempt."
                )
                self.real.operation.finished_at = time.monotonic()
                self._persist_private_session(best_effort=True)

    def _record_progress(self, step: int, total_steps: int) -> None:
        with self._lock:
            self.real.operation.step = step
            self.real.operation.total_steps = total_steps
            self.real.operation.phase = "training"

    def focused_local_use(
        self,
        *,
        candidate_key: str,
        prompt: str,
        maximum_tokens: int = 64,
        seed: int = 17,
        save: bool = True,
    ) -> dict[str, object]:
        if self.mode == "fixture_demo":
            fixture_result = self.demo.focused_local_use(
                candidate_key=candidate_key,
                prompt=prompt,
                maximum_tokens=maximum_tokens,
                seed=seed,
                save=save,
            )
            return {
                **fixture_result,
                "mode": "fixture_demo",
                "demo": True,
                "artifact_label": "Fixture demo payload - not a trained adapter",
                "inference_label": "Synthetic fixture response - no model inference",
            }
        if self.mode != "real_local":
            raise ConsumerJourneyError("consumer_setup_required")
        state = self._real_use_context(candidate_key)
        assert state.result is not None and state.result.artifact is not None
        assert state.backend is not None and state.sources is not None
        assert state.adapter is not None and state.model is not None
        assert state.group is not None and state.target is not None
        runtime = LibraryInferenceRuntime(
            state.backend, state.sources, state.adapter.runtime_identity
        )
        try:
            result = LocalUseService(self.project_root, runtime=runtime).focused(
                LocalUseRequest(
                    state.result.artifact,
                    state.model,
                    state.group,
                    state.target,
                    InferenceSettings(
                        temperature=0,
                        maximum_tokens=maximum_tokens,
                        seed=seed,
                    ),
                    ({"text": prompt},),
                    _new_attempt_id("session-real-consumer") if save else None,
                )
            )
        except LibraryRuntimeError as exc:
            raise ConsumerJourneyError(
                exc.code,
                {"action": "Verify the local model, tokenizer, and adapter bytes."},
            ) from None
        view = result.to_view()
        view.update(
            {
                "mode": "real_local",
                "demo": False,
                "artifact_label": "Verified real trained LoRA adapter",
                "inference_label": "Real local model inference",
                "candidate_key": "selected",
                "focused_local_use": True,
                "general_chat": False,
            }
        )
        self.real.local_result = view
        return view

    def batch_local_use(
        self,
        *,
        candidate_key: str,
        prompts: tuple[str, ...],
        maximum_tokens: int = 64,
        seed: int = 17,
        save: bool = False,
    ) -> dict[str, object]:
        if self.mode == "fixture_demo":
            fixture_result = self.demo.batch_local_use(
                candidate_key=candidate_key,
                prompts=prompts,
                maximum_tokens=maximum_tokens,
                seed=seed,
                save=save,
            )
            return {
                **fixture_result,
                "mode": "fixture_demo",
                "demo": True,
                "artifact_label": "Fixture demo payload - not a trained adapter",
                "inference_label": "Synthetic fixture response - no model inference",
            }
        if self.mode != "real_local":
            raise ConsumerJourneyError("consumer_setup_required")
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ConsumerJourneyError("local_batch_inputs_invalid")
        state = self._real_use_context(candidate_key)
        assert state.result is not None and state.result.artifact is not None
        assert state.backend is not None and state.sources is not None
        assert state.adapter is not None and state.model is not None
        assert state.group is not None and state.target is not None
        runtime = LibraryInferenceRuntime(
            state.backend, state.sources, state.adapter.runtime_identity
        )
        try:
            result = LocalUseService(self.project_root, runtime=runtime).batch(
                LocalUseRequest(
                    state.result.artifact,
                    state.model,
                    state.group,
                    state.target,
                    InferenceSettings(0, maximum_tokens, seed),
                    tuple({"text": prompt} for prompt in prompts),
                    _new_attempt_id("session-real-batch") if save else None,
                )
            )
        except LibraryRuntimeError as exc:
            raise ConsumerJourneyError(
                exc.code,
                {"action": "Verify the local model, tokenizer, and adapter bytes."},
            ) from None
        view = result.to_view()
        view.update(
            {
                "mode": "real_local",
                "demo": False,
                "artifact_label": "Verified real trained LoRA adapter",
                "inference_label": "Real local model inference",
                "candidate_key": "selected",
                "batch_size": len(prompts),
                "general_chat": False,
            }
        )
        self.real.local_result = view
        return view

    def export_selected(self, *, candidate_key: str) -> dict[str, object]:
        if self.mode == "fixture_demo":
            fixture_result = self.demo.export_selected(candidate_key=candidate_key)
            return {
                **fixture_result,
                "mode": "fixture_demo",
                "demo": True,
                "artifact_label": "Fixture demo payload - not a trained adapter",
                "export_label": (
                    "Fixture demo payload manifest - not a trained adapter"
                ),
            }
        if self.mode != "real_local":
            raise ConsumerJourneyError("consumer_setup_required")
        state = self._real_use_context(candidate_key)
        assert state.result is not None and state.result.artifact is not None
        assert state.model is not None and state.group is not None
        assert state.target is not None
        exported = LocalUseService(self.project_root).export(
            AdapterExportRequest(
                _new_attempt_id("export-real-consumer"),
                state.result.artifact,
                state.model,
                state.group,
                state.target,
            )
        )
        view = exported.to_view()
        view.update(
            {
                "mode": "real_local",
                "demo": False,
                "artifact_label": "Verified real trained LoRA adapter",
                "export_label": "Portable real LoRA adapter bundle",
            }
        )
        self.real.local_result = view
        return view

    def _ensure_no_active_run(self, code: str, action: str) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ConsumerJourneyError(code, {"action": action})

    def _real_use_context(self, candidate_key: str) -> RealState:
        if candidate_key != "selected":
            raise ConsumerJourneyError(
                "real_candidate_invalid",
                {"action": "Use the single verified real candidate."},
            )
        state = self.real
        if (
            state.result is None
            or state.result.status is not RunLifecycleStatus.COMPLETED
            or state.result.artifact is None
            or state.backend is None
            or state.sources is None
            or state.adapter is None
            or state.model is None
            or state.group is None
            or state.target is None
        ):
            raise ConsumerJourneyError(
                "real_adapter_not_ready",
                {
                    "action": (
                        "Complete a real run and wait for verified adapter integrity."
                    )
                },
            )
        return state

    @property
    def _session_path(self) -> Path:
        return self.project_root / CONSUMER_SESSION_RELATIVE_PATH

    def _restore_private_session(self) -> None:
        if not self._session_path.exists():
            return
        try:
            value = loads_canonical_json(read_stable_bytes(self._session_path))
        except (SafeIoError, OSError, TypeError, ValueError):
            return
        if (
            not isinstance(value, dict)
            or value.get("schema") != CONSUMER_SESSION_SCHEMA
        ):
            return
        mode = value.get("mode")
        if mode == "fixture_demo":
            self.mode = mode
            return
        if mode != "real_local":
            return
        self.mode = mode
        setup_value = value.get("setup")
        setup: RealSetup | None = None
        failure_code = "restart_reimport_required"
        if isinstance(setup_value, dict) and all(
            isinstance(setup_value.get(field), str) and setup_value[field]
            for field in (
                "model_source",
                "tokenizer_source",
                "display_name",
                "model_family",
                "architecture",
                "revision",
                "license",
                "target",
            )
        ):
            model_source = Path(setup_value["model_source"])
            tokenizer_source = Path(setup_value["tokenizer_source"])
            if (
                model_source.is_absolute()
                and tokenizer_source.is_absolute()
                and model_source.is_dir()
                and tokenizer_source.is_dir()
            ):
                setup = RealSetup(
                    model_source.resolve(),
                    tokenizer_source.resolve(),
                    setup_value["display_name"],
                    setup_value["model_family"],
                    setup_value["architecture"],
                    setup_value["revision"],
                    setup_value["license"],
                    setup_value["target"],
                )
            else:
                failure_code = "restart_local_sources_unavailable"
        self.real = RealState(
            setup=setup,
            recovery_required=True,
            operation=OperationState(
                status="interrupted",
                phase="restart_recovery_required",
                failure_code=failure_code,
                recovery_action=(
                    "Reconfirm local sources, re-import the dataset, and resolve a "
                    "new run attempt. No prior artifact is represented as restored."
                ),
            ),
        )

    def _persist_private_session(self, *, best_effort: bool = False) -> None:
        setup = self.real.setup
        payload: dict[str, object] = {
            "schema": CONSUMER_SESSION_SCHEMA,
            "mode": self.mode,
            "operation": {
                "status": self.real.operation.status,
                "phase": self.real.operation.phase,
            },
        }
        if self.mode == "real_local" and setup is not None:
            payload["setup"] = {
                "model_source": str(setup.model_source),
                "tokenizer_source": str(setup.tokenizer_source),
                "display_name": setup.display_name,
                "model_family": setup.model_family,
                "architecture": setup.architecture,
                "revision": setup.revision,
                "license": setup.license,
                "target": setup.target,
            }
        try:
            replace_bytes(self._session_path, dumps_canonical_json(payload))
        except (SafeIoError, OSError, TypeError, ValueError):
            if not best_effort:
                raise ConsumerJourneyError(
                    "consumer_session_state_unavailable",
                    {
                        "action": (
                            "Choose a writable local project directory and retry setup."
                        )
                    },
                ) from None

    def reconcile_pending_operations(self) -> dict[str, object]:
        """Reconcile only the workflow selected by the private session marker."""

        if self.mode == "fixture_demo":
            return self.demo.reconcile_pending_operations()
        if self.mode == "real_local":
            return {
                "status": "restart_recovery_required"
                if self.real.recovery_required
                else "no_pending_operation",
                "operation": self.real.operation.to_view(),
            }
        return {"status": "not_configured"}

    def preview_cleanup(self, entry_ids: tuple[str, ...]) -> dict[str, object]:
        if self.mode == "fixture_demo":
            return self.demo.preview_cleanup(entry_ids)
        raise self._fixture_only_error()

    def execute_cleanup(
        self,
        plan_id: str,
        *,
        confirm: bool,
        entry_ids: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        if self.mode == "fixture_demo":
            return self.demo.execute_cleanup(
                plan_id, confirm=confirm, entry_ids=entry_ids
            )
        raise self._fixture_only_error()

    def prepare_replay(self, candidate_key: str, mode: str) -> dict[str, object]:
        if self.mode == "fixture_demo":
            return self.demo.prepare_replay(candidate_key, mode)
        raise self._fixture_only_error()

    def execute_replay(
        self,
        plan_id: str,
        *,
        run_id: str,
        candidate_key: str | None = None,
        mode: str | None = None,
    ) -> dict[str, object]:
        if self.mode == "fixture_demo":
            return self.demo.execute_replay(
                plan_id,
                run_id=run_id,
                candidate_key=candidate_key,
                mode=mode,
            )
        raise self._fixture_only_error()

    def _fixture_only_error(self) -> ConsumerJourneyError:
        if self.mode == "real_local":
            return ConsumerJourneyError(
                "real_action_not_available",
                {
                    "action": (
                        "This fixture-only control is not part of the real local "
                        "training path."
                    )
                },
            )
        return ConsumerJourneyError("consumer_setup_required")

    def workspace(self) -> dict[str, object]:
        if self.mode is None:
            return _unconfigured_workspace()
        if self.mode != "real_local":
            value = self.demo.workspace()
            candidates = tuple(self.demo.state.candidates)
            if candidates and not value.get("resolutions"):
                value["resolutions"] = [
                    candidate.resolution.to_payload() for candidate in candidates
                ]
                stages = value.get("stages")
                if isinstance(stages, list):
                    for stage in stages:
                        if isinstance(stage, dict) and stage.get("key") == "recipe":
                            stage["complete"] = True
                            stage["state"] = "complete"
            value.update(
                {
                    "mode": "fixture_demo",
                    "mode_label": "Fixture demo - no model is trained",
                    "demo": True,
                    "artifact_label": "Deterministic fixture payload",
                    "inference_label": "Synthetic fixture response",
                }
            )
            artifact_values = value.get("artifacts", [])
            if isinstance(artifact_values, list):
                for artifact in artifact_values:
                    if isinstance(artifact, dict):
                        artifact.update(
                            {
                                "demo": True,
                                "artifact_kind": "fixture_demo_payload",
                                "label": (
                                    "Fixture demo payload - not a trained adapter"
                                ),
                            }
                        )
            return value
        with self._lock:
            result = self.real.result
            artifacts: list[dict[str, object]] = []
            runs: list[dict[str, object]] = []
            if result is not None:
                run = result.to_view()
                run.update({"demo": False, "runtime_kind": "library"})
                runs.append(run)
                if (
                    result.status is RunLifecycleStatus.COMPLETED
                    and result.artifact is not None
                ):
                    artifacts.append(
                        {
                            "key": "selected",
                            "reference": record_reference(result.artifact).to_dict(),
                            "demo": False,
                            "artifact_kind": "real_trained_lora_adapter",
                            "label": "Verified real trained LoRA adapter",
                            "integrity_status": "verified",
                            "available": True,
                        }
                    )
            setup_done = self.real.setup is not None
            data_done = self.real.prepared is not None
            resolved = self.real.launch is not None
            trained = (
                result is not None and result.status is RunLifecycleStatus.COMPLETED
            )
            active_write = self.real.operation.status == "running"
            if active_write:
                summary = self.real.store_snapshot
                if summary is None:
                    summary = (
                        {
                            "status": "write_in_progress",
                            "record_count": 0,
                            "event_count": 0,
                            "bundle_manifest_count": 0,
                        },
                        [],
                        [],
                    )
            else:
                summary = _real_store_summary(self.project_root)
                self.real.store_snapshot = summary
            store, saved_sessions, verified_exports = summary
            store = dict(store)
            if active_write:
                store.update(
                    {
                        "status": "write_in_progress",
                        "snapshot_status": store.get("status", "unknown"),
                        "snapshot_during_active_write": True,
                    }
                )
            return {
                "mode": "real_local",
                "mode_label": "Real local training - library-backed",
                "demo": False,
                "project": self.real.opened.to_view()
                if self.real.opened
                else ({"status": "configured"} if setup_done else None),
                "dataset": self.real.dataset_view,
                "resolutions": [self.real.resolution.to_payload()]
                if self.real.resolution
                else [],
                "runs": runs,
                "artifacts": artifacts,
                "operation": self.real.operation.to_view(),
                "recovery_required": self.real.recovery_required,
                "local_result": self.real.local_result,
                "capability": self.real.capability.to_public_facts()
                if self.real.capability
                else None,
                "selected_target": self.real.setup.target if self.real.setup else None,
                "stages": [
                    {"key": "setup", "complete": setup_done, "applicable": True},
                    {"key": "data", "complete": data_done, "applicable": True},
                    {"key": "recipe", "complete": resolved, "applicable": True},
                    {"key": "run", "complete": trained, "applicable": True},
                    {"key": "evaluate", "complete": False, "applicable": False},
                    {
                        "key": "use",
                        "complete": self.real.local_result is not None,
                        "applicable": True,
                    },
                    {"key": "storage", "complete": False, "applicable": False},
                ],
                "recommendation": {
                    "selected_candidate": "selected" if trained else None,
                    "confidence": "single-real-candidate",
                },
                "registry": [
                    {
                        "key": "selected",
                        "current": True,
                        "label": "Automatic single real candidate",
                    }
                ]
                if trained
                else [],
                "retention": {
                    "entries": [],
                    "entry_count": 0,
                    "logical_bytes": 0,
                    "physical_bytes": 0,
                    "reclaimable_physical_bytes": 0,
                    "receipts": [],
                    "available": False,
                },
                "reproduction": {"active_plan": None, "executions": []},
                "store": store,
                "local_use": {
                    "saved_session_count": len(saved_sessions),
                    "export_count": len(verified_exports),
                },
                "records": [],
                "events": [],
                "reviews": [],
                "decisions": [],
                "saved_sessions": saved_sessions,
                "verified_exports": verified_exports,
            }

    def __getattr__(self, name: str) -> Any:
        if name == "_replay_draft":
            return getattr(self.demo, name)
        if name.startswith("_"):
            raise AttributeError(name)
        if self.mode == "real_local":
            raise ConsumerJourneyError(
                "real_action_not_available",
                {
                    "action": (
                        "Complete the real training and local-use path shown in the "
                        "workspace."
                    )
                },
            )
        if self.mode is None:
            raise ConsumerJourneyError("consumer_setup_required")
        return getattr(self.demo, name)

    def _require_real_setup(self) -> RealSetup:
        if self.mode != "real_local" or self.real.setup is None:
            raise ConsumerJourneyError("real_setup_required")
        return self.real.setup

    def _real_dataset_context(
        self,
    ) -> tuple[RealSetup, OpenedProject, BaseModelRevision, PreparedDataset]:
        setup = self._require_real_setup()
        if (
            self.real.opened is None
            or self.real.model is None
            or self.real.prepared is None
        ):
            raise ConsumerJourneyError("real_dataset_required")
        return setup, self.real.opened, self.real.model, self.real.prepared


def _new_attempt_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:16]}"


def _bounded_preview(
    value: Mapping[str, object], *, maximum_characters: int = 2_000
) -> dict[str, object]:
    result = dict(value)
    text = result.get("text")
    if isinstance(text, str) and len(text) > maximum_characters:
        result["text"] = f"{text[:maximum_characters]}\n[preview truncated]"
        result["text_truncated"] = True
        result["full_text_characters"] = len(text)
    else:
        result["text_truncated"] = False
    return result


def _real_store_summary(
    project_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if not (project_root / ".temper").is_dir():
        return (
            {
                "status": "not_created",
                "record_count": 0,
                "event_count": 0,
                "bundle_count": 0,
            },
            [],
            [],
        )
    store = TypedEvidenceStore(project_root)
    verification = store.verify().to_dict()
    records = tuple(item.record for item in store.iter_records())
    sessions = [
        record_reference(item).to_dict()
        for item in records
        if isinstance(item, LocalUseSession)
    ]
    exports = [
        record_reference(item).to_dict()
        for item in records
        if isinstance(item, AdapterExport)
    ]
    return verification, sessions, exports


def _unconfigured_workspace() -> dict[str, object]:
    return {
        "mode": "unconfigured",
        "mode_label": "Choose fixture demo or real local training",
        "demo": False,
        "project": None,
        "dataset": None,
        "resolutions": [],
        "runs": [],
        "artifacts": [],
        "operation": OperationState().to_view(),
        "local_result": None,
        "stages": [
            {"key": key, "complete": False, "applicable": True}
            for key in ("setup", "data", "recipe", "run", "evaluate", "use", "storage")
        ],
        "recommendation": {"selected_candidate": None, "confidence": "none"},
        "registry": [],
        "retention": {
            "entries": [],
            "entry_count": 0,
            "logical_bytes": 0,
            "physical_bytes": 0,
            "reclaimable_physical_bytes": 0,
            "receipts": [],
        },
        "reproduction": {"active_plan": None, "executions": []},
        "store": {"status": "empty", "record_count": 0, "event_count": 0},
        "local_use": {"saved_session_count": 0, "export_count": 0},
        "records": [],
        "events": [],
        "reviews": [],
        "decisions": [],
        "saved_sessions": [],
        "verified_exports": [],
    }


def _library_recovery_action(code: str, target: str) -> str:
    if code.endswith("_unavailable"):
        return (
            "Install the Temper runtime extra in the selected execution environment "
            "and retry preflight."
        )
    if code in {
        "library_model_source_invalid",
        "library_tokenizer_source_invalid",
        "library_sources_preflight_failed",
        "library_target_modules_preflight_failed",
    }:
        return (
            "Choose complete local model and tokenizer directories compatible with "
            "Transformers, then retry."
        )
    if code == "library_target_modules_unsupported":
        return "Choose LoRA target module names present in the selected local model."
    if target == "wsl_rocm":
        return (
            "Verify the explicit WSL distribution, worker Python, mapped sources, "
            "and ROCm runtime. Native CPU fallback is disabled."
        )
    return "Verify the selected local backend, hardware, model, and tokenizer."


def _create_real_project(
    project_root: Path,
    setup: RealSetup,
    rendering_contract: ContentIdentity,
    weights_identity: ContentIdentity,
    tokenizer_identity: ContentIdentity,
    attempt: str,
) -> tuple[OpenedProject, BaseModelRevision]:
    task = TaskDefinition(
        f"task-real-consumer-{attempt}",
        "Local trace completion",
        "Train a local LoRA adapter from explicitly mapped trace data.",
        {"required": ["context"]},
        {"required": ["completion"]},
        rendering_contract,
        ("local_trace_completion",),
        ("text_generation",),
    )
    model = BaseModelRevision(
        f"model-real-consumer-{attempt}",
        setup.display_name,
        setup.model_family,
        setup.architecture,
        "local-import",
        setup.revision,
        weights_identity,
        tokenizer_identity,
        setup.license,
    )
    project = Project(
        f"project-real-consumer-{attempt}",
        "Real local adapter project",
        "Train and verify one local LoRA adapter through Temper.",
        record_reference(task),
        (record_reference(model),),
    )
    baseline = BaselinePolicy(
        f"baseline-real-consumer-{attempt}",
        (PerModelBaseline(_identity("baseline-policy")),),
    )
    policy = ProjectPolicy(
        f"policy-real-consumer-{attempt}",
        record_reference(project),
        record_reference(task),
        rendering_contract,
        _identity("evaluation-policy"),
        (_identity("case-suite"),),
        _identity("readiness-policy"),
        _identity("retention-policy"),
        ("local-lora",),
        record_reference(baseline),
        _identity("recommendation-policy"),
    )
    opened = ProjectService(project_root).create(
        ProjectCreateRequest(task, project, baseline, policy, (model,))
    )
    return opened, model


def _default_backend(setup: RealSetup) -> LibraryBackend:
    if setup.target == "native_local":
        return TransformersPeftBackend()
    required = {
        "distribution": "TEMPER_SLICE8_WSL_DISTRIBUTION",
        "worker_python": "TEMPER_SLICE8_WORKER_PYTHON",
        "host_staging": "TEMPER_SLICE8_HOST_STAGING_ROOT",
        "worker_staging": "TEMPER_SLICE8_WORKER_STAGING_ROOT",
        "worker_model": "TEMPER_SLICE8_WORKER_MODEL_SOURCE",
        "worker_tokenizer": "TEMPER_SLICE8_WORKER_TOKENIZER_SOURCE",
    }
    missing = [
        field for field, variable in required.items() if not os.environ.get(variable)
    ]
    if missing:
        raise ConsumerJourneyError(
            "wsl_rocm_configuration_incomplete",
            {
                "missing_fields": missing,
                "action": (
                    "Configure the explicit WSL worker paths; Temper will not use "
                    "native CPU as a silent fallback."
                ),
            },
        )
    config = WslWorkerConfig(
        target_class="wsl_rocm",
        launch=WslWorkerLaunchSpec(
            os.environ[required["distribution"]],
            PurePosixPath(os.environ[required["worker_python"]]),
            timeout_seconds=900,
        ),
        path_map=WindowsWslPathMap(
            PureWindowsPath(os.environ[required["host_staging"]]),
            PurePosixPath(os.environ[required["worker_staging"]]),
        ),
        host_model_source=setup.model_source,
        host_tokenizer_source=setup.tokenizer_source,
        worker_model_source=PurePosixPath(os.environ[required["worker_model"]]),
        worker_tokenizer_source=PurePosixPath(os.environ[required["worker_tokenizer"]]),
    )
    return WslWorkerBackend(config)


def _directory_identity(path: Path) -> tuple[ContentIdentity, int]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ConsumerJourneyError("local_source_empty")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                total += len(chunk)
                digest.update(chunk)
    return ContentIdentity("sha256", digest.hexdigest()), total


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode("utf-8")).hexdigest())


def _local_directory(value: str, code: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ConsumerJourneyError(code)
    return path.resolve()


def _adapter_for_format(source_format: str):
    return {
        "json": JsonDatasetAdapter,
        "jsonl": JsonlDatasetAdapter,
        "csv": CsvDatasetAdapter,
    }.get(source_format, lambda: _raise_format())()


def _raise_format():
    raise ConsumerJourneyError(
        "dataset_format_unsupported",
        {"supported_formats": ["json", "jsonl", "csv", "hugging_face"]},
    )


def _parse_hugging_face_dataset_url(value: str) -> tuple[str, str | None]:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ConsumerJourneyError("hugging_face_url_invalid")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ConsumerJourneyError("hugging_face_url_invalid")
    dataset_id = f"{parts[1]}/{parts[2]}"
    file_path = None
    if len(parts) > 5 and parts[3] in {"blob", "resolve"}:
        file_path = "/".join(parts[5:])
    return dataset_id, file_path


def _read_public_url(url: str, maximum_bytes: int) -> bytes:
    try:
        request = Request(url, headers={"User-Agent": "Temper-ML/0.1 local-import"})
        with urlopen(request, timeout=60) as response:
            final_host = urlsplit(response.geturl()).hostname or ""
            if not (
                final_host == "huggingface.co"
                or final_host.endswith(".huggingface.co")
                or final_host.endswith(".hf.co")
            ):
                raise ConsumerJourneyError("hugging_face_redirect_invalid")
            data = response.read(maximum_bytes + 1)
    except ConsumerJourneyError:
        raise
    except Exception:
        raise ConsumerJourneyError(
            "hugging_face_download_failed",
            {
                "action": (
                    "Verify the public dataset URL, config, split, or file and retry."
                )
            },
        ) from None
    if len(data) > maximum_bytes:
        raise ConsumerJourneyError(
            "hugging_face_download_too_large",
            {
                "maximum_bytes": maximum_bytes,
                "action": "Use config/split row limiting or a local file.",
            },
        )
    return data


def _require_trainable_analysis(analysis: DatasetAnalysis) -> None:
    if analysis.accepted_rows == 0:
        raise DatasetPreflightError("dataset_has_no_accepted_rows", analysis)
    counts = {item.split: item.count for item in analysis.split_counts}
    if counts.get("train", 0) == 0:
        raise DatasetPreflightError("dataset_required_split_empty", analysis)


def _limit_imported_source(
    imported: ImportedSource, row_limit: int
) -> tuple[ImportedSource, int]:
    """Bind a bounded pilot to the exact selected row prefix."""
    if row_limit < 1 or row_limit > MAX_HUGGING_FACE_ROWS:
        raise ConsumerJourneyError("dataset_row_limit_invalid")
    available = len(imported.rows)
    if available <= row_limit:
        return imported, available
    return HuggingFaceRowsDatasetAdapter().load(imported.rows[:row_limit]), available


def _text_option(options: Mapping[str, object], name: str, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or not value.strip():
        if default == "":
            return ""
        raise ConsumerJourneyError("request_option_invalid", {"field": name})
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ConsumerJourneyError("request_option_invalid")
    return value.strip() or None


def _optional_field(value: object) -> str | None:
    return _optional_text(value)


def _int_option(
    options: Mapping[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConsumerJourneyError("request_option_invalid", {"field": name})
    if maximum is not None and value > maximum:
        raise ConsumerJourneyError("request_option_invalid", {"field": name})
    return value


def _optional_positive_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ConsumerJourneyError("request_option_invalid")
    if not isinstance(value, (int, str)):
        raise ConsumerJourneyError("request_option_invalid")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ConsumerJourneyError("request_option_invalid") from None
    if result < 1:
        raise ConsumerJourneyError("request_option_invalid")
    return result


def _decimal_option(
    options: Mapping[str, object],
    name: str,
    default: str,
    *,
    minimum: Decimal,
    maximum_exclusive: Decimal | None = None,
) -> Decimal:
    value = options.get(name, default)
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ConsumerJourneyError("request_option_invalid", {"field": name})
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ConsumerJourneyError("request_option_invalid", {"field": name}) from None
    if (
        not result.is_finite()
        or result < minimum
        or (maximum_exclusive is not None and result >= maximum_exclusive)
    ):
        raise ConsumerJourneyError("request_option_invalid", {"field": name})
    return result
