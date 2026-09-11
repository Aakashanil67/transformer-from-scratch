import torch

from transformer_lab.experiments.runtime import environment_metadata, peak_memory


def test_runtime_metadata_is_safe_on_cpu() -> None:
    metadata = environment_metadata(torch.device("cpu"))
    memory = peak_memory(torch.device("cpu"))

    assert metadata["device"] == "cpu"
    assert metadata["cuda_available"] is False
    assert {"numpy", "torch", "scikit-learn", "transformers", "joblib"} <= set(metadata["packages"])
    assert memory["peak_cuda_allocated_bytes"] is None
