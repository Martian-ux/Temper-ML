from decimal import Decimal
import hashlib
import io
from pathlib import Path, PurePosixPath, PureWindowsPath
import os

import pytest
import temper_ml.runtime.library_backend as library_backend_module

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import RecordEnvelope, RecordReference, record_reference
from temper_ml.runtime.fixture_inference import InferenceSettings
from temper_ml.runtime.library_backend import (
    LibraryExecutionContext,
    LibraryRuntimeError,
    TransformersPeftBackend,
)
from temper_ml.runtime.paths import WindowsWslPathMap
from temper_ml.runtime.protocol import RuntimeOperation
from temper_ml.runtime.worker_port import WslWorkerLaunchSpec
from temper_ml.runtime.wsl_backend import WslWorkerBackend, WslWorkerConfig
from temper_ml.store.canonical_json import (
    dumps_canonical_json,
    loads_canonical_json,
)

HARDWARE_REQUEST_PROJECTION = HashProjection("runtime.hardware_test_request", "v1")


class _UnsupportedCheckpointGlobal:
    pass


def _synthetic_reference(record_type: str, logical_id: str) -> RecordReference:
    digest = hashlib.sha256(f"{record_type}:{logical_id}".encode()).hexdigest()
    return RecordReference(
        record_type,
        logical_id,
        ContentIdentity("sha256", digest),
    )


def _resume_resolution() -> RecipeResolution:
    return RecipeResolution(
        resolution_id="resolution-real-resume-regression",
        recipe=_synthetic_reference("recipe", "recipe-real-resume-regression"),
        base_model_revision=_synthetic_reference(
            "base_model_revision", "model-real-resume-regression"
        ),
        hardware_requirements=_synthetic_reference(
            "hardware_requirements", "requirements-real-resume-regression"
        ),
        execution_target=_synthetic_reference(
            "execution_target", "target-real-resume-regression"
        ),
        adapter_type="lora",
        target_modules=("c_attn",),
        rank=2,
        alpha=4,
        dropout=Decimal("0.2"),
        learning_rate=Decimal("0.001"),
        effective_batch_size=2,
        sequence_length=8,
        optimizer="adamw",
        precision="fp32",
        gradient_accumulation=2,
        seed=17,
        schedule="linear",
        training_steps=4,
        checkpoint_cadence=2,
        quantization="none",
        library_versions={
            "accelerate": "runtime-test",
            "peft": "runtime-test",
            "torch": "runtime-test",
            "transformers": "runtime-test",
        },
        applied_constraints=(),
    )


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"unmet Slice 8 hardware capability: {name} is not configured")
    return value


@pytest.mark.hardware
def test_wsl_rocm_probe_train_and_infer_are_capability_gated() -> None:
    if os.environ.get("TEMPER_RUN_SLICE8_HARDWARE") != "1":
        pytest.skip(
            "unmet Slice 8 hardware capability: explicit hardware opt-in is disabled"
        )
    resolution_path = Path(_required_environment("TEMPER_SLICE8_RESOLUTION"))
    value = loads_canonical_json(resolution_path.read_bytes())
    if not isinstance(value, dict):
        pytest.fail("Slice 8 hardware resolution is not a record envelope")
    record = RecordEnvelope.from_dict(value).to_record()
    if not isinstance(record, RecipeResolution):
        pytest.fail("Slice 8 hardware resolution has the wrong record type")
    if record.training_steps > 2:
        pytest.skip(
            "unmet Slice 8 hardware capability: smoke resolution exceeds two steps"
        )
    target_class = _required_environment("TEMPER_SLICE8_TARGET_CLASS")
    config = WslWorkerConfig(
        target_class=target_class,
        launch=WslWorkerLaunchSpec(
            _required_environment("TEMPER_SLICE8_WSL_DISTRIBUTION"),
            PurePosixPath(_required_environment("TEMPER_SLICE8_WORKER_PYTHON")),
            timeout_seconds=900,
        ),
        path_map=WindowsWslPathMap(
            PureWindowsPath(_required_environment("TEMPER_SLICE8_HOST_STAGING_ROOT")),
            PurePosixPath(_required_environment("TEMPER_SLICE8_WORKER_STAGING_ROOT")),
        ),
        host_model_source=Path(
            _required_environment("TEMPER_SLICE8_HOST_MODEL_SOURCE")
        ),
        host_tokenizer_source=Path(
            _required_environment("TEMPER_SLICE8_HOST_TOKENIZER_SOURCE")
        ),
        worker_model_source=PurePosixPath(
            _required_environment("TEMPER_SLICE8_WORKER_MODEL_SOURCE")
        ),
        worker_tokenizer_source=PurePosixPath(
            _required_environment("TEMPER_SLICE8_WORKER_TOKENIZER_SOURCE")
        ),
    )
    backend = WslWorkerBackend(config)
    capability = backend.probe()
    if capability.accelerator_backend != "rocm" or capability.accelerator_count < 1:
        pytest.skip("unmet Slice 8 hardware capability: ROCm accelerator unavailable")
    request_identity = content_identity(
        HARDWARE_REQUEST_PROJECTION,
        {
            "schema_version": "v1",
            "target_class": target_class,
            "recipe_resolution": record_reference(record).to_dict(),
        },
    )
    context = LibraryExecutionContext(
        request_identity,
        "slice8-hardware-smoke",
        RuntimeOperation.TRAIN,
        target_class,
    )
    progress: list[tuple[int, int]] = []
    checkpoints = []
    result = backend.train(
        context=context,
        model_source=config.host_model_source,
        tokenizer_source=config.host_tokenizer_source,
        rendered_dataset=(
            dumps_canonical_json({"text": "Synthetic hardware smoke input"})
        ),
        resolution=record,
        resume_checkpoint=None,
        on_progress=lambda step, loss: progress.append((step, loss)),
        on_checkpoint=checkpoints.append,
        on_heartbeat=lambda step: None,
        cancellation_requested=lambda: False,
        interruption_requested=lambda: False,
    )
    assert result.adapter_payload is not None
    assert result.adapter_payload_format is not None
    assert progress
    inference = backend.infer(
        context=LibraryExecutionContext(
            request_identity,
            "slice8-hardware-inference",
            RuntimeOperation.EVALUATE,
            target_class,
        ),
        model_source=config.host_model_source,
        tokenizer_source=config.host_tokenizer_source,
        adapter_payload=result.adapter_payload,
        adapter_payload_format=result.adapter_payload_format,
        resolution=record,
        settings=InferenceSettings(),
        inputs=("Synthetic hardware evaluation input",),
    )
    assert len(inference.outputs) == 1


def test_real_backend_resume_matches_uninterrupted_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("accelerate")

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(
            self,
            text,
            *,
            truncation,
            max_length,
            padding,
            return_tensors,
        ):
            del truncation, padding, return_tensors
            tokens = [2 + (ord(character) % 29) for character in text][:max_length]
            attention = [1] * len(tokens)
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
            attention.extend([0] * (max_length - len(attention)))
            return {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
                "attention_mask": torch.tensor([attention], dtype=torch.long),
            }

    def model_factory(*args, **kwargs):
        del args, kwargs
        config = transformers.GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_ctx=16,
            n_embd=16,
            n_layer=1,
            n_head=1,
            resid_pdrop=0.2,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=1,
            use_cache=False,
        )
        return transformers.GPT2LMHeadModel(config)

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: TinyTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        model_factory,
    )
    model_source = tmp_path / "synthetic-model"
    tokenizer_source = tmp_path / "synthetic-tokenizer"
    model_source.mkdir()
    tokenizer_source.mkdir()
    resolution = _resume_resolution()
    request_identity = content_identity(
        HARDWARE_REQUEST_PROJECTION,
        {
            "schema_version": "v1",
            "test_case": "real-resume-regression",
        },
    )
    rendered_dataset = b"".join(
        dumps_canonical_json({"text": text})
        for text in ("Synthetic alpha", "Synthetic beta", "Synthetic gamma")
    )

    def train(run_id, *, resume_checkpoint=None, interrupt=None):
        observed_checkpoints = []

        def interruption_requested():
            return bool(
                interrupt
                and observed_checkpoints
                and observed_checkpoints[-1].step == interrupt
            )

        result = TransformersPeftBackend().train(
            context=LibraryExecutionContext(
                request_identity,
                run_id,
                RuntimeOperation.TRAIN,
                "native_cpu",
            ),
            model_source=model_source,
            tokenizer_source=tokenizer_source,
            rendered_dataset=rendered_dataset,
            resolution=resolution,
            resume_checkpoint=resume_checkpoint,
            on_progress=lambda step, loss: None,
            on_checkpoint=observed_checkpoints.append,
            on_heartbeat=lambda step: None,
            cancellation_requested=lambda: False,
            interruption_requested=interruption_requested,
        )
        return result

    uninterrupted = train("run-real-uninterrupted")
    interrupted = train("run-real-interrupted", interrupt=2)
    checkpoint = interrupted.checkpoints[-1]
    recovered = train(
        "run-real-recovered",
        resume_checkpoint=checkpoint.payload,
    )

    assert interrupted.interrupted is True
    assert checkpoint.step == 2
    checkpoint_state = torch.load(
        io.BytesIO(checkpoint.payload), map_location="cpu", weights_only=True
    )
    assert checkpoint_state["batches_consumed"] == 3
    assert interrupted.progress + recovered.progress == uninterrupted.progress
    assert recovered.adapter_config == uninterrupted.adapter_config
    assert recovered.adapter_payload == uninterrupted.adapter_payload


def test_real_backend_normalizes_malformed_checkpoint_payloads() -> None:
    torch = pytest.importorskip("torch")
    resolution = _resume_resolution()
    valid = io.BytesIO()
    torch.save({"schema_version": "v2"}, valid)
    unsupported = io.BytesIO()
    torch.save(_UnsupportedCheckpointGlobal(), unsupported)

    for payload in (
        b"not-a-torch-checkpoint",
        valid.getvalue()[:-10],
        unsupported.getvalue(),
    ):
        with pytest.raises(
            LibraryRuntimeError, match="library_checkpoint_restore_failed"
        ):
            library_backend_module._checkpoint_state(torch, payload, resolution)
