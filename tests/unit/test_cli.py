import pytest

from transformer_lab.cli import build_parser, main


def test_train_lm_command_requires_an_explicit_config_path() -> None:
    parser = build_parser()

    args = parser.parse_args(["train-lm", "--config", "configs/lm/bigram.toml"])

    assert args.command == "train-lm"
    assert args.config == "configs/lm/bigram.toml"


def test_train_lm_command_exposes_resume_without_changing_the_output_path() -> None:
    parser = build_parser()

    args = parser.parse_args(["train-lm", "--config", "configs/lm/bigram.toml", "--resume"])

    assert args.resume is True


def test_train_sentiment_command_exposes_resume() -> None:
    args = build_parser().parse_args(
        ["train-sentiment", "--config", "configs/sentiment/lora.toml", "--resume"]
    )

    assert args.resume is True


def test_sentiment_matrix_command_accepts_configs_and_seeds() -> None:
    args = build_parser().parse_args(
        [
            "run-sentiment-matrix",
            "--config",
            "baseline.toml",
            "--config",
            "lora.toml",
            "--seeds",
            "17",
            "23",
            "41",
            "--device",
            "cuda",
        ]
    )

    assert args.config == ["baseline.toml", "lora.toml"]
    assert args.seeds == [17, 23, 41]


def test_selection_and_frozen_evaluation_commands_require_their_manifests() -> None:
    selection = build_parser().parse_args(
        ["select-sentiment", "--protocol", "configs/sentiment/v3-protocol.toml"]
    )
    evaluation = build_parser().parse_args(
        [
            "evaluate-matrix",
            "--selection",
            "reports/results/v3/selection.json",
            "--seeds",
            "17",
            "23",
            "41",
            "--device",
            "cpu",
            "--output",
            "comparison.json",
        ]
    )

    assert selection.command == "select-sentiment"
    assert evaluation.command == "evaluate-matrix"
    assert evaluation.seeds == [17, 23, 41]


def test_candidate_training_command_requires_validation_summary_path() -> None:
    args = build_parser().parse_args(
        [
            "train-sentiment-candidate",
            "--config",
            "candidate.toml",
            "--summary",
            "summary.json",
            "--device",
            "cuda",
            "--seed",
            "17",
        ]
    )

    assert args.command == "train-sentiment-candidate"
    assert args.summary == "summary.json"
    assert args.seed == 17


@pytest.mark.parametrize(
    "command",
    ["sample", "verify-gpt2", "train-sentiment", "evaluate-sentiment"],
)
def test_each_public_command_accepts_a_config_path(command: str) -> None:
    parser = build_parser()

    args = parser.parse_args([command, "--config", "configs/example.toml"])

    assert args.command == command
    assert args.config == "configs/example.toml"


def test_module_help_lists_the_training_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])

    assert "train-lm" in capsys.readouterr().out


def test_main_dispatches_a_training_command(monkeypatch, tmp_path, capsys) -> None:
    expected = tmp_path / "result.json"
    monkeypatch.setattr("transformer_lab.cli.train_lm", lambda *args, **kwargs: expected)

    assert main(["train-lm", "--config", "config.toml", "--device", "cpu"]) == 0
    assert str(expected) in capsys.readouterr().out


def test_main_reports_a_domain_error_without_a_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "transformer_lab.cli.sample",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing checkpoint")),
    )

    assert main(["sample", "--config", "config.toml"]) == 2
    assert "missing checkpoint" in capsys.readouterr().err


def test_main_dispatches_selection(monkeypatch, tmp_path, capsys) -> None:
    expected = tmp_path / "selection.json"
    monkeypatch.setattr("transformer_lab.cli.select_sentiment", lambda *args, **kwargs: expected)

    assert main(["select-sentiment", "--protocol", "protocol.toml"]) == 0
    assert str(expected) in capsys.readouterr().out
