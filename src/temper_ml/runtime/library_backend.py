"""Optional PyTorch/Transformers/PEFT/Accelerate runtime machinery."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import Decimal
import importlib
from importlib import metadata
import io
import os
from pathlib import Path
import pickle
import platform
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import (
    RecordValidationError,
    freeze_json_object,
    record_reference,
    require_identifier,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.fixture_inference import InferenceSettings
from temper_ml.runtime.protocol import RuntimeOperation
from temper_ml.runtime.staging import TransferReceipt
from temper_ml.store.canonical_json import loads_canonical_json


class LibraryRuntimeError(RuntimeError):
    """A stable failure that does not disclose model data or local paths."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class LibraryExecutionContext:
    """Portable subject facts for one backend operation."""

    request_identity: ContentIdentity
    run_id: str
    operation: RuntimeOperation
    target_class: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_identity, ContentIdentity) or not isinstance(
            self.operation, RuntimeOperation
        ):
            raise LibraryRuntimeError("library_execution_context_invalid")
        try:
            require_identifier("run_id", self.run_id)
            require_identifier("target_class", self.target_class)
        except RecordValidationError:
            raise LibraryRuntimeError("library_execution_context_invalid") from None


@dataclass(frozen=True)
class LibraryCapability:
    """Public-safe facts observed inside one concrete worker environment."""

    accelerator_backend: str
    accelerator_architecture: str
    accelerator_model: str
    accelerator_count: int
    accelerator_memory_bytes: tuple[int, ...]
    system_memory_bytes: int
    supported_precision_modes: tuple[str, ...]
    supported_quantization_modes: tuple[str, ...]
    capabilities: tuple[str, ...]
    library_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        try:
            require_identifier("accelerator_backend", self.accelerator_backend)
        except RecordValidationError:
            raise LibraryRuntimeError("library_capability_invalid") from None
        if not self.accelerator_architecture or not self.accelerator_model:
            raise LibraryRuntimeError("library_capability_invalid")
        if not _public_text(self.accelerator_architecture) or not _public_text(
            self.accelerator_model
        ):
            raise LibraryRuntimeError("library_capability_invalid")
        if (
            isinstance(self.accelerator_count, bool)
            or not isinstance(self.accelerator_count, int)
            or self.accelerator_count < 0
            or len(self.accelerator_memory_bytes) != self.accelerator_count
        ):
            raise LibraryRuntimeError("library_capability_invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.accelerator_memory_bytes
        ):
            raise LibraryRuntimeError("library_capability_invalid")
        if (
            isinstance(self.system_memory_bytes, bool)
            or not isinstance(self.system_memory_bytes, int)
            or self.system_memory_bytes < 1
        ):
            raise LibraryRuntimeError("library_capability_invalid")
        for values in (
            self.supported_precision_modes,
            self.supported_quantization_modes,
            self.capabilities,
        ):
            if (
                not isinstance(values, tuple)
                or tuple(sorted(values)) != values
                or len(set(values)) != len(values)
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise LibraryRuntimeError("library_capability_invalid")
        if not isinstance(self.library_versions, Mapping) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in self.library_versions.items()
        ):
            raise LibraryRuntimeError("library_capability_invalid")
        object.__setattr__(
            self,
            "library_versions",
            MappingProxyType(dict(sorted(self.library_versions.items()))),
        )

    def to_public_facts(self) -> dict[str, object]:
        return {
            "accelerator_backend": self.accelerator_backend,
            "accelerator_architecture": self.accelerator_architecture,
            "accelerator_model": self.accelerator_model,
            "accelerator_count": self.accelerator_count,
            "accelerator_memory_bytes": list(self.accelerator_memory_bytes),
            "system_memory_bytes": self.system_memory_bytes,
            "supported_precision_modes": list(self.supported_precision_modes),
            "supported_quantization_modes": list(self.supported_quantization_modes),
            "capabilities": list(self.capabilities),
            "library_versions": dict(sorted(self.library_versions.items())),
        }


@dataclass(frozen=True)
class LibraryCheckpointPayload:
    step: int
    payload: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 1
            or not isinstance(self.payload, bytes)
            or not self.payload
        ):
            raise LibraryRuntimeError("library_checkpoint_invalid")


@dataclass(frozen=True)
class LibraryTrainingResult:
    adapter_payload: bytes | None
    adapter_payload_format: str | None
    adapter_config: Mapping[str, object] | None
    progress: tuple[tuple[int, int], ...]
    checkpoints: tuple[LibraryCheckpointPayload, ...]
    cancelled: bool = False
    interrupted: bool = False
    disconnected: bool = False
    transport_receipts: tuple[TransferReceipt, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cancelled, bool)
            or not isinstance(self.interrupted, bool)
            or not isinstance(self.disconnected, bool)
            or (self.disconnected and not self.interrupted)
        ):
            raise LibraryRuntimeError("library_training_result_invalid")
        if self.cancelled and self.interrupted:
            raise LibraryRuntimeError("library_training_result_invalid")
        terminal_without_artifact = self.cancelled or self.interrupted
        if terminal_without_artifact != (self.adapter_payload is None):
            raise LibraryRuntimeError("library_training_result_invalid")
        if terminal_without_artifact != (self.adapter_payload_format is None):
            raise LibraryRuntimeError("library_training_result_invalid")
        if terminal_without_artifact != (self.adapter_config is None):
            raise LibraryRuntimeError("library_training_result_invalid")
        if self.adapter_payload is not None and not self.adapter_payload:
            raise LibraryRuntimeError("library_training_result_invalid")
        if self.adapter_payload_format is not None:
            try:
                require_identifier(
                    "adapter_payload_format", self.adapter_payload_format
                )
            except RecordValidationError:
                raise LibraryRuntimeError("library_training_result_invalid") from None
            if self.adapter_payload_format != "safetensors":
                raise LibraryRuntimeError("library_training_result_invalid")
        if self.adapter_config is not None:
            try:
                frozen = freeze_json_object(
                    self.adapter_config, field="library_training_result.adapter_config"
                )
            except (RecordValidationError, TypeError, ValueError):
                raise LibraryRuntimeError("library_training_result_invalid") from None
            object.__setattr__(self, "adapter_config", frozen)
        if not isinstance(self.progress, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or isinstance(item[0], bool)
            or not isinstance(item[0], int)
            or item[0] < 1
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            for item in self.progress
        ):
            raise LibraryRuntimeError("library_training_result_invalid")
        progress_steps = tuple(item[0] for item in self.progress)
        if tuple(sorted(progress_steps)) != progress_steps or len(
            set(progress_steps)
        ) != len(progress_steps):
            raise LibraryRuntimeError("library_training_result_invalid")
        if not isinstance(self.checkpoints, tuple) or any(
            not isinstance(item, LibraryCheckpointPayload) for item in self.checkpoints
        ):
            raise LibraryRuntimeError("library_training_result_invalid")
        checkpoint_steps = tuple(item.step for item in self.checkpoints)
        if tuple(sorted(checkpoint_steps)) != checkpoint_steps or len(
            set(checkpoint_steps)
        ) != len(checkpoint_steps):
            raise LibraryRuntimeError("library_training_result_invalid")
        if not isinstance(self.transport_receipts, tuple) or any(
            not isinstance(receipt, TransferReceipt)
            for receipt in self.transport_receipts
        ):
            raise LibraryRuntimeError("library_training_result_invalid")


@dataclass(frozen=True)
class LibraryInferenceResult:
    outputs: tuple[str, ...]
    library_versions: Mapping[str, str]
    transport_receipts: tuple[TransferReceipt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, tuple) or not self.outputs:
            raise LibraryRuntimeError("library_inference_result_invalid")
        if any(not isinstance(value, str) for value in self.outputs):
            raise LibraryRuntimeError("library_inference_result_invalid")
        if not isinstance(self.library_versions, Mapping) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or not _public_text(version)
            for name, version in self.library_versions.items()
        ):
            raise LibraryRuntimeError("library_inference_result_invalid")
        object.__setattr__(
            self,
            "library_versions",
            MappingProxyType(dict(sorted(self.library_versions.items()))),
        )
        if not isinstance(self.transport_receipts, tuple) or any(
            not isinstance(receipt, TransferReceipt)
            for receipt in self.transport_receipts
        ):
            raise LibraryRuntimeError("library_inference_result_invalid")


ProgressCallback = Callable[[int, int], None]
CheckpointCallback = Callable[[LibraryCheckpointPayload], None]
HeartbeatCallback = Callable[[int], None]
CancellationCheck = Callable[[], bool]
InterruptionCheck = Callable[[], bool]


@runtime_checkable
class LibraryBackend(Protocol):
    """The narrow library seam implemented by real and deterministic backends."""

    def probe(self) -> LibraryCapability: ...

    def train(
        self,
        *,
        context: LibraryExecutionContext,
        model_source: Path,
        tokenizer_source: Path,
        rendered_dataset: bytes,
        resolution: RecipeResolution,
        resume_checkpoint: bytes | None,
        on_progress: ProgressCallback,
        on_checkpoint: CheckpointCallback,
        on_heartbeat: HeartbeatCallback,
        cancellation_requested: CancellationCheck,
        interruption_requested: InterruptionCheck,
    ) -> LibraryTrainingResult: ...

    def infer(
        self,
        *,
        context: LibraryExecutionContext,
        model_source: Path,
        tokenizer_source: Path,
        adapter_payload: bytes,
        adapter_payload_format: str,
        resolution: RecipeResolution,
        settings: InferenceSettings,
        inputs: tuple[str, ...],
    ) -> LibraryInferenceResult: ...


class TransformersPeftBackend:
    """A local-only LoRA implementation using the supported library stack."""

    def probe(self) -> LibraryCapability:
        torch = _required_module("torch")
        _required_module("transformers")
        _required_module("peft")
        _required_module("accelerate")
        versions = _library_versions()
        cuda = getattr(torch, "cuda", None)
        count = (
            int(cuda.device_count()) if cuda is not None and cuda.is_available() else 0
        )
        memories: list[int] = []
        models: list[str] = []
        architecture = "cpu"
        backend = "cpu"
        if count and cuda is not None:
            hip = getattr(getattr(torch, "version", object()), "hip", None)
            backend = "rocm" if hip else "cuda"
            architecture = "amd-gpu" if hip else "nvidia-gpu"
            for index in range(count):
                properties = cuda.get_device_properties(index)
                memories.append(int(properties.total_memory))
                models.append(_public_device_name(str(properties.name)))
        precision = ["fp32"]
        if count:
            precision.append("fp16")
            is_bf16_supported = getattr(cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported) and is_bf16_supported():
                precision.append("bf16")
        quantization = ["none"]
        if _optional_module_available("bitsandbytes"):
            quantization.extend(("int4", "int8"))
        capabilities = [
            "accelerate",
            "cancellation",
            "checkpoint_resume",
            "evaluation_inference",
            "local_staging",
            "local_use_inference",
            "lora",
            "peft",
            "transformers",
        ]
        if backend == "rocm":
            capabilities.append("rocm")
        return LibraryCapability(
            accelerator_backend=backend,
            accelerator_architecture=architecture,
            accelerator_model=(models[0] if models else "cpu-runtime"),
            accelerator_count=count,
            accelerator_memory_bytes=tuple(memories),
            system_memory_bytes=_system_memory_bytes(),
            supported_precision_modes=tuple(sorted(precision)),
            supported_quantization_modes=tuple(sorted(quantization)),
            capabilities=tuple(sorted(capabilities)),
            library_versions=versions,
        )

    def train(
        self,
        *,
        context: LibraryExecutionContext,
        model_source: Path,
        tokenizer_source: Path,
        rendered_dataset: bytes,
        resolution: RecipeResolution,
        resume_checkpoint: bytes | None,
        on_progress: ProgressCallback,
        on_checkpoint: CheckpointCallback,
        on_heartbeat: HeartbeatCallback,
        cancellation_requested: CancellationCheck,
        interruption_requested: InterruptionCheck,
    ) -> LibraryTrainingResult:
        if (
            not isinstance(context, LibraryExecutionContext)
            or context.operation is not RuntimeOperation.TRAIN
        ):
            raise LibraryRuntimeError("library_execution_context_invalid")
        _require_local_directory(model_source, "library_model_source_invalid")
        _require_local_directory(tokenizer_source, "library_tokenizer_source_invalid")
        if not isinstance(resolution, RecipeResolution):
            raise LibraryRuntimeError("library_resolution_invalid")
        if resolution.adapter_type.casefold() != "lora":
            raise LibraryRuntimeError("library_adapter_type_unsupported")
        texts = _rendered_texts(rendered_dataset)
        torch = _required_module("torch")
        transformers = _required_module("transformers")
        peft = _required_module("peft")
        accelerate = _required_module("accelerate")
        transformers.set_seed(resolution.seed)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(tokenizer_source), local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise LibraryRuntimeError("library_tokenizer_padding_unsupported")
            tokenizer.pad_token = tokenizer.eos_token
        load_options = _model_load_options(torch, transformers, resolution)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_source),
            local_files_only=True,
            trust_remote_code=False,
            **load_options,
        )
        lora_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=resolution.rank,
            lora_alpha=resolution.alpha,
            lora_dropout=float(resolution.dropout),
            target_modules=list(resolution.target_modules),
            bias="none",
        )
        model = peft.get_peft_model(model, lora_config)
        samples = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=resolution.sequence_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        batch_size = max(
            1,
            resolution.effective_batch_size // resolution.gradient_accumulation,
        )
        loader = torch.utils.data.DataLoader(
            samples, batch_size=batch_size, shuffle=False
        )
        if resolution.optimizer not in {"adamw", "adamw_torch"}:
            raise LibraryRuntimeError("library_optimizer_unsupported")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(resolution.learning_rate),
        )
        scheduler = transformers.get_scheduler(
            resolution.schedule,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=resolution.training_steps,
        )
        checkpoint_state = _checkpoint_state(torch, resume_checkpoint, resolution)
        mixed_precision = {
            "fp16": "fp16",
            "bf16": "bf16",
        }.get(resolution.precision, "no")
        accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=resolution.gradient_accumulation,
            mixed_precision=mixed_precision,
        )
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )
        batches = _repeat_loader(loader)
        if checkpoint_state is not None:
            _validate_checkpoint_loader_position(checkpoint_state, loader, resolution)
            for _ in range(int(checkpoint_state["batches_consumed"])):
                next(batches)
            _restore_checkpoint(
                torch,
                peft,
                accelerator.unwrap_model(model),
                optimizer,
                scheduler,
                accelerator,
                checkpoint_state,
            )
        model.train()
        start_step = (
            int(checkpoint_state["step"]) if checkpoint_state is not None else 0
        )
        batches_consumed = (
            int(checkpoint_state["batches_consumed"])
            if checkpoint_state is not None
            else 0
        )
        completed_steps = start_step
        progress: list[tuple[int, int]] = []
        checkpoints: list[LibraryCheckpointPayload] = []
        for batch in batches:
            if interruption_requested():
                return LibraryTrainingResult(
                    None,
                    None,
                    None,
                    tuple(progress),
                    tuple(checkpoints),
                    interrupted=True,
                )
            if cancellation_requested():
                return LibraryTrainingResult(
                    None,
                    None,
                    None,
                    tuple(progress),
                    tuple(checkpoints),
                    cancelled=True,
                )
            batches_consumed += 1
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if not accelerator.sync_gradients:
                continue
            completed_steps += 1
            loss_microunits = int(round(float(loss.detach().item()) * 1_000_000))
            progress.append((completed_steps, loss_microunits))
            on_progress(completed_steps, loss_microunits)
            on_heartbeat(completed_steps)
            checkpoint_due = (
                completed_steps % resolution.checkpoint_cadence == 0
                or completed_steps == resolution.training_steps
            )
            if checkpoint_due:
                checkpoint = LibraryCheckpointPayload(
                    completed_steps,
                    _checkpoint_bytes(
                        torch,
                        peft,
                        accelerator.unwrap_model(model),
                        optimizer,
                        scheduler,
                        accelerator,
                        completed_steps,
                        batches_consumed,
                        resolution,
                    ),
                )
                checkpoints.append(checkpoint)
                on_checkpoint(checkpoint)
            if interruption_requested():
                return LibraryTrainingResult(
                    None,
                    None,
                    None,
                    tuple(progress),
                    tuple(checkpoints),
                    interrupted=True,
                )
            if completed_steps >= resolution.training_steps:
                break
        unwrapped = accelerator.unwrap_model(model)
        with tempfile.TemporaryDirectory(prefix="temper-adapter-") as directory:
            unwrapped.save_pretrained(directory, safe_serialization=True)
            root = Path(directory)
            safetensors = root / "adapter_model.safetensors"
            pytorch_bin = root / "adapter_model.bin"
            if not safetensors.is_file():
                raise LibraryRuntimeError("library_adapter_output_missing")
            if pytorch_bin.is_file():
                raise LibraryRuntimeError("library_adapter_output_unsafe")
            payload = safetensors.read_bytes()
            payload_format = "safetensors"
        config: dict[str, object] = {
            "peft_type": "lora",
            "task_type": "causal_lm",
            "bias": "none",
            "rank": resolution.rank,
            "alpha": resolution.alpha,
            "dropout": _decimal_text(resolution.dropout),
            "payload_format": payload_format,
        }
        return LibraryTrainingResult(
            payload,
            payload_format,
            config,
            tuple(progress),
            tuple(checkpoints),
        )

    def infer(
        self,
        *,
        context: LibraryExecutionContext,
        model_source: Path,
        tokenizer_source: Path,
        adapter_payload: bytes,
        adapter_payload_format: str,
        resolution: RecipeResolution,
        settings: InferenceSettings,
        inputs: tuple[str, ...],
    ) -> LibraryInferenceResult:
        if not isinstance(
            context, LibraryExecutionContext
        ) or context.operation not in {
            RuntimeOperation.EVALUATE,
            RuntimeOperation.INFER_FOCUSED,
            RuntimeOperation.INFER_BATCH,
        }:
            raise LibraryRuntimeError("library_execution_context_invalid")
        _require_local_directory(model_source, "library_model_source_invalid")
        _require_local_directory(tokenizer_source, "library_tokenizer_source_invalid")
        if not adapter_payload or not inputs:
            raise LibraryRuntimeError("library_inference_input_invalid")
        if adapter_payload_format != "safetensors":
            raise LibraryRuntimeError("library_inference_config_invalid")
        torch = _required_module("torch")
        transformers = _required_module("transformers")
        peft = _required_module("peft")
        accelerate = _required_module("accelerate")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(tokenizer_source), local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_source),
            local_files_only=True,
            trust_remote_code=False,
            **_model_load_options(torch, transformers, resolution),
        )
        with tempfile.TemporaryDirectory(prefix="temper-inference-") as directory:
            root = Path(directory)
            config = {
                "base_model_name_or_path": str(model_source),
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "lora_alpha": resolution.alpha,
                "lora_dropout": float(resolution.dropout),
                "peft_type": "LORA",
                "r": resolution.rank,
                "target_modules": list(resolution.target_modules),
                "task_type": "CAUSAL_LM",
            }
            import json

            (root / "adapter_config.json").write_text(
                json.dumps(config, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            (root / "adapter_model.safetensors").write_bytes(adapter_payload)
            model = peft.PeftModel.from_pretrained(model, str(root), is_trainable=False)
            accelerator = accelerate.Accelerator()
            model = accelerator.prepare_model(model, evaluation_mode=True)
            device = accelerator.device
            outputs: list[str] = []
            generator = torch.Generator(device=device)
            generator.manual_seed(settings.seed)
            for text in inputs:
                encoded = tokenizer(text, return_tensors="pt").to(device)
                generate_options: dict[str, object] = {
                    "max_new_tokens": settings.maximum_tokens,
                    "do_sample": settings.temperature != 0,
                    "generator": generator,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if settings.temperature != 0:
                    generate_options["temperature"] = float(settings.temperature)
                generated = model.generate(**encoded, **generate_options)
                prompt_tokens = int(encoded["input_ids"].shape[-1])
                outputs.append(
                    tokenizer.decode(
                        generated[0, prompt_tokens:], skip_special_tokens=True
                    )
                )
        return LibraryInferenceResult(tuple(outputs), _library_versions())


def _required_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        raise LibraryRuntimeError(f"library_{name}_unavailable") from None


def _optional_module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except (ImportError, OSError):
        return False


def _library_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in (
        "accelerate",
        "bitsandbytes",
        "peft",
        "safetensors",
        "torch",
        "transformers",
    ):
        try:
            result[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return dict(sorted(result.items()))


def _public_device_name(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized[:128] if normalized else "accelerator"


def _system_memory_bytes() -> int:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            windll: Any = getattr(ctypes, "windll", None)
            kernel32: Any = getattr(windll, "kernel32", None)
            global_memory_status = getattr(kernel32, "GlobalMemoryStatusEx", None)
            if callable(global_memory_status) and global_memory_status(
                ctypes.byref(status)
            ):
                return int(status.total_physical)
        sysconf = getattr(os, "sysconf", None)
        if not callable(sysconf):
            return 1
        page_size = int(sysconf("SC_PAGE_SIZE"))
        pages = int(sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except (AttributeError, OSError, TypeError, ValueError):
        return 1


def _rendered_texts(data: bytes) -> tuple[str, ...]:
    if not isinstance(data, bytes) or not data:
        raise LibraryRuntimeError("library_dataset_invalid")
    values: list[str] = []
    for raw in data.splitlines(keepends=True):
        try:
            item = loads_canonical_json(raw)
        except (TypeError, ValueError):
            raise LibraryRuntimeError("library_dataset_invalid") from None
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise LibraryRuntimeError("library_dataset_invalid")
        values.append(item["text"])
    if not values:
        raise LibraryRuntimeError("library_dataset_invalid")
    return tuple(values)


def _require_local_directory(path: Path, code: str) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or not path.is_dir():
        raise LibraryRuntimeError(code)


def _model_load_options(
    torch: Any, transformers: Any, resolution: RecipeResolution
) -> dict[str, object]:
    options: dict[str, object] = {}
    dtype = {
        "fp16": getattr(torch, "float16"),
        "bf16": getattr(torch, "bfloat16"),
        "fp32": getattr(torch, "float32"),
    }.get(resolution.precision)
    if dtype is None:
        raise LibraryRuntimeError("library_precision_unsupported")
    options["torch_dtype"] = dtype
    if resolution.quantization == "none":
        return options
    bits = getattr(transformers, "BitsAndBytesConfig", None)
    if bits is None or not _optional_module_available("bitsandbytes"):
        raise LibraryRuntimeError("library_quantization_unavailable")
    if resolution.quantization == "int4":
        options["quantization_config"] = bits(load_in_4bit=True)
    elif resolution.quantization == "int8":
        options["quantization_config"] = bits(load_in_8bit=True)
    else:
        raise LibraryRuntimeError("library_quantization_unsupported")
    return options


def _checkpoint_bytes(
    torch: Any,
    peft: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    accelerator: Any,
    step: int,
    batches_consumed: int,
    resolution: RecipeResolution,
) -> bytes:
    buffer = io.BytesIO()
    torch_rng_state, cuda_rng_states = _capture_torch_rng_state(torch)
    scaler = getattr(accelerator, "scaler", None)
    state = {
        "schema_version": "v2",
        "step": step,
        "batches_consumed": batches_consumed,
        "recipe_resolution": record_reference(resolution).to_dict(),
        "adapter_state": peft.get_peft_model_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "torch_rng_state": torch_rng_state,
        "cuda_rng_states": cuda_rng_states,
        "accelerator_scaler_state": (
            scaler.state_dict() if scaler is not None else None
        ),
    }
    torch.save(state, buffer)
    return buffer.getvalue()


def _restore_checkpoint(
    torch: Any,
    peft: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    accelerator: Any,
    state: Mapping[str, Any],
) -> None:
    try:
        peft.set_peft_model_state_dict(model, state["adapter_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler = getattr(accelerator, "scaler", None)
        scaler_state = state["accelerator_scaler_state"]
        if (scaler is None) != (scaler_state is None):
            raise ValueError
        if scaler is not None:
            scaler.load_state_dict(scaler_state)
        _restore_torch_rng_state(
            torch,
            state["torch_rng_state"],
            tuple(state["cuda_rng_states"]),
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        raise LibraryRuntimeError("library_checkpoint_restore_failed") from None


def _checkpoint_state(
    torch: Any,
    payload: bytes | None,
    resolution: RecipeResolution,
) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    try:
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "step",
            "batches_consumed",
            "recipe_resolution",
            "adapter_state",
            "optimizer_state",
            "scheduler_state",
            "torch_rng_state",
            "cuda_rng_states",
            "accelerator_scaler_state",
        }:
            raise ValueError
        step = value["step"]
        batches_consumed = value["batches_consumed"]
        if (
            value["schema_version"] != "v2"
            or value["recipe_resolution"] != record_reference(resolution).to_dict()
            or isinstance(step, bool)
            or not isinstance(step, int)
            or not 1 <= step < resolution.training_steps
            or isinstance(batches_consumed, bool)
            or not isinstance(batches_consumed, int)
            or batches_consumed < step
            or batches_consumed > step * resolution.gradient_accumulation
            or not isinstance(value["adapter_state"], Mapping)
            or not isinstance(value["optimizer_state"], Mapping)
            or not isinstance(value["scheduler_state"], Mapping)
            or not _is_torch_rng_state(torch, value["torch_rng_state"])
            or not isinstance(value["cuda_rng_states"], (list, tuple))
            or any(
                not _is_torch_rng_state(torch, item)
                for item in value["cuda_rng_states"]
            )
            or (
                value["accelerator_scaler_state"] is not None
                and not isinstance(value["accelerator_scaler_state"], Mapping)
            )
        ):
            raise ValueError
    except (
        EOFError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise LibraryRuntimeError("library_checkpoint_restore_failed") from None
    return value


def _validate_checkpoint_loader_position(
    state: Mapping[str, Any], loader: Any, resolution: RecipeResolution
) -> None:
    try:
        loader_batches = len(loader)
        step = state["step"]
        batches_consumed = state["batches_consumed"]
        accumulation = resolution.gradient_accumulation
        if loader_batches < 1:
            raise ValueError
        steps_per_loader = (loader_batches + accumulation - 1) // accumulation
        completed_loaders, steps_in_loader = divmod(step, steps_per_loader)
        expected = completed_loaders * loader_batches + steps_in_loader * accumulation
        if batches_consumed != expected:
            raise ValueError
    except (KeyError, OverflowError, TypeError, ValueError):
        raise LibraryRuntimeError("library_checkpoint_restore_failed") from None


def _capture_torch_rng_state(torch: Any) -> tuple[Any, tuple[Any, ...]]:
    cpu_state = torch.get_rng_state()
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return cpu_state, ()
    return cpu_state, tuple(cuda.get_rng_state_all())


def _restore_torch_rng_state(
    torch: Any, cpu_state: Any, cuda_states: tuple[Any, ...]
) -> None:
    torch.set_rng_state(cpu_state)
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        if cuda_states:
            raise ValueError
        return
    if len(cuda_states) != int(cuda.device_count()):
        raise ValueError
    cuda.set_rng_state_all(cuda_states)


def _is_torch_rng_state(torch: Any, value: Any) -> bool:
    is_tensor = getattr(torch, "is_tensor", None)
    return callable(is_tensor) and bool(is_tensor(value))


def _repeat_loader(loader: Any) -> Iterator[Any]:
    """Repeat a prepared loader without caching all accelerator batches."""

    while True:
        yield from loader


def _decimal_text(value: Decimal | int) -> str:
    return str(value)


def _public_text(value: str) -> bool:
    return bool(
        value
        and not value.startswith(("/", "\\\\"))
        and "://" not in value
        and not (
            len(value) >= 3
            and value[0].isalpha()
            and value[1] == ":"
            and value[2] in {"/", "\\"}
        )
    )
