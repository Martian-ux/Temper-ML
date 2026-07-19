from pathlib import Path, PurePosixPath, PureWindowsPath
import os

import pytest

from temper_ml.domain.projections import HashProjection, content_identity
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import RecordEnvelope, record_reference
from temper_ml.runtime.fixture_inference import InferenceSettings
from temper_ml.runtime.library_backend import LibraryExecutionContext
from temper_ml.runtime.paths import WindowsWslPathMap
from temper_ml.runtime.protocol import RuntimeOperation
from temper_ml.runtime.worker_port import WslWorkerLaunchSpec
from temper_ml.runtime.wsl_backend import WslWorkerBackend, WslWorkerConfig
from temper_ml.store.canonical_json import (
    dumps_canonical_json,
    loads_canonical_json,
)

HARDWARE_REQUEST_PROJECTION = HashProjection("runtime.hardware_test_request", "v1")


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
            dumps_canonical_json({"text": "Synthetic hardware smoke input"}) + b"\n"
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
