import torch

from transformer_lab.experiments.runtime import (
    environment_metadata,
    file_sha256,
    peak_memory,
    source_provenance,
)


def test_runtime_metadata_is_safe_on_cpu() -> None:
    metadata = environment_metadata(torch.device("cpu"))
    memory = peak_memory(torch.device("cpu"))

    assert metadata["device"] == "cpu"
    assert metadata["cuda_available"] is False
    assert {"numpy", "torch", "scikit-learn", "transformers", "joblib"} <= set(metadata["packages"])
    assert memory["peak_cuda_allocated_bytes"] is None


def test_source_provenance_changes_when_public_source_changes(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    package = tmp_path / "src" / "fixture"
    package.mkdir(parents=True)
    module = package / "model.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    config.write_text("seed = 17\n", encoding="utf-8")

    first = source_provenance(config)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    second = source_provenance(config)

    assert first["source_tree_sha256"] != second["source_tree_sha256"]
    assert first["config_sha256"] == file_sha256(config)
    assert first["git_commit"] is None
    assert first["git_dirty"] is None
