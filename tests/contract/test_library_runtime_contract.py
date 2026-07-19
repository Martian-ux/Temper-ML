from pathlib import PurePosixPath, PureWindowsPath

from temper_ml.runtime.library_backend import (
    LibraryBackend,
    LibraryCapability,
    TransformersPeftBackend,
)
from temper_ml.runtime.library_double import DeterministicLibraryBackend
from temper_ml.runtime.paths import WindowsWslPathMap
from temper_ml.runtime.protocol import RuntimeOperation
from temper_ml.runtime.worker_port import WslWorkerLaunchSpec
from temper_ml.runtime.wsl_backend import WslWorkerBackend, WslWorkerConfig


def test_every_slice8_backend_implements_one_narrow_runtime_port(tmp_path) -> None:
    capability = LibraryCapability(
        accelerator_backend="cpu",
        accelerator_architecture="synthetic-cpu",
        accelerator_model="Synthetic CPU",
        accelerator_count=0,
        accelerator_memory_bytes=(),
        system_memory_bytes=1,
        supported_precision_modes=("fp32",),
        supported_quantization_modes=("none",),
        capabilities=("lora",),
        library_versions={
            "accelerate": "1.test",
            "peft": "1.test",
            "torch": "1.test",
            "transformers": "1.test",
        },
    )
    wsl = WslWorkerBackend(
        WslWorkerConfig(
            target_class="wsl_rocm",
            launch=WslWorkerLaunchSpec(
                "Ubuntu-ROCm", PurePosixPath("/usr/bin/python3")
            ),
            path_map=WindowsWslPathMap(
                PureWindowsPath(str(tmp_path)), PurePosixPath("/temper-staging")
            ),
            host_model_source=(tmp_path / "model").resolve(),
            host_tokenizer_source=(tmp_path / "tokenizer").resolve(),
            worker_model_source=PurePosixPath("/models/base"),
            worker_tokenizer_source=PurePosixPath("/models/tokenizer"),
        )
    )

    assert isinstance(DeterministicLibraryBackend(capability), LibraryBackend)
    assert isinstance(TransformersPeftBackend(), LibraryBackend)
    assert isinstance(wsl, LibraryBackend)
    assert tuple(operation.value for operation in RuntimeOperation) == (
        "probe",
        "train",
        "evaluate",
        "infer_focused",
        "infer_batch",
    )
