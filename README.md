# Transformer from scratch

This repository implements a compact decoder-only transformer in raw PyTorch and then checks its computations against GPT-2. It also provides a byte-level BPE for study, a reproducible Financial PhraseBank reference experiment, low-rank adapter components, and a local Streamlit playground.

Keep three claims separate: the architecture is implemented from scratch; a small byte-level model is trained from random initialisation; and a separate local model converts pinned pretrained GPT-2 weights. The parity and sentiment results belong to the converted GPT-2 path, not to the scratch-training demonstration.

The project is deliberately inspectable. The important operations are in `src/transformer_lab`, rather than hidden behind a training framework.

## The problem

The project asks whether a small, readable transformer can reproduce a public GPT-2 checkpoint closely enough to support genuine generation and downstream classification. A toy language-model loss cannot answer that on its own. The repository therefore checks the forward pass against pinned GPT-2 weights, compares parameter-efficient sentiment updates with full tuning and a TF-IDF reference, and keeps the artefacts required to replay each score.

## Verified historical results (fixed profile)

| Check | Result |
| --- | --- |
| GPT-2 small conversion | 5 fixed prompts; max absolute logit error `1.125e-4` at a `2e-4` CPU tolerance; next-token agreement 100% |
| Expanded GPT-2 parity | 4 random token lengths, one mixed-length batch and all hidden-state boundaries; maximum hidden-state error `4.883e-4` at a `1e-3` tolerance |
| Financial PhraseBank TF-IDF reference | Test macro-F1 `0.8007` (95% bootstrap interval `0.7547–0.8414`); accuracy `0.8533` |
| Financial PhraseBank GPT-2 head-only | Three-seed test macro-F1 `0.4136 ± 0.0227`; accuracy `0.6918 ± 0.0150` |
| Financial PhraseBank GPT-2 LoRA | Three-seed test macro-F1 `0.6931 ± 0.0471`; accuracy `0.7960 ± 0.0175` |
| Financial PhraseBank GPT-2 full tuning | Three-seed test macro-F1 `0.8687 ± 0.0037`; accuracy `0.8996 ± 0.0039` |
| Tiny Shakespeare byte-bigram | Validation loss `5.990` → `2.772` over 100 updates |
| Tiny Shakespeare scratch transformer | Validation loss `5.558` → `2.162` over 1,000 optimiser updates; 842,496 parameters |

The result summaries are tracked in [`reports/results`](reports/results). Raw data, pretrained weights, checkpoints, and transient run files are ignored. The GPT-2 parity summary uses the pinned `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The Financial PhraseBank archive uses dataset revision `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`.

## Exploratory v3 follow-up (validation-selected)

The current comparison table is [`reports/generated-results-v3.md`](reports/generated-results-v3.md). Its settings were selected from validation-only candidate runs and frozen in the [selection manifest](reports/results/v3/selection.json) before seeds 17, 23, and 41 were evaluated. The manifest, [12 candidate summaries](reports/results/v3/candidates), and replayable [per-seed records](reports/results/v3/runs) are retained with the table. The follow-up reuses the historical test partition, so it is exploratory rather than an independent holdout; it should not be read as a benchmark record or a general method ranking.

## Install

Python 3.11 or 3.12 is supported.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -c requirements\constraints-cpu.txt -e ".[all]"
```

The base package uses CPU/CUDA-agnostic PyTorch requirements. For the optional GTX 1650 setup, follow [`docs/gpu-setup.md`](docs/gpu-setup.md) and verify CUDA before selecting `--device cuda`.

## Test and lint

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff format --check src tests scripts app
.venv\Scripts\python.exe -m ruff check src tests scripts app
```

## Run the examples

Prepare Tiny Shakespeare, train the small byte-bigram, and sample from the local checkpoint:

```powershell
.venv\Scripts\python.exe scripts\prepare_tiny_shakespeare.py
.venv\Scripts\transformer-lab.exe train-lm --config configs\lm\tiny-shakespeare.toml --device cpu
.venv\Scripts\transformer-lab.exe sample --config configs\lm\tiny-shakespeare.toml --device cpu --seed 17
```

Check the raw transformer against the pinned GPT-2 checkpoint:

```powershell
.venv\Scripts\transformer-lab.exe verify-gpt2 --config configs\gpt2\parity.toml --device cpu
```

Train the implemented decoder from random initialisation on the prepared byte corpus, then sample it:

```powershell
.venv\Scripts\transformer-lab.exe train-lm --config configs\lm\tiny-shakespeare-transformer.toml --device cpu --seed 17
.venv\Scripts\transformer-lab.exe sample --config configs\lm\tiny-shakespeare-transformer.toml --device cpu --seed 17
```

Run the Financial PhraseBank reference model:

```powershell
.venv\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\baseline.toml --device cpu
.venv\Scripts\transformer-lab.exe evaluate-sentiment --config configs\sentiment\baseline.toml --device cpu
```

The GPT-2 sentiment profiles share the same duplicate-safe split and metric code. The recorded historical runs used the documented CUDA 12.8 environment on a GeForce GTX 1650:

```powershell
.venv-gpu\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\head-only-gpt2-small.toml --device cuda
.venv-gpu\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\lora-gpt2-small.toml --device cuda
.venv-gpu\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\full-gpt2-small.toml --device cuda
.venv-gpu\Scripts\transformer-lab.exe evaluate-sentiment --config configs\sentiment\lora-gpt2-small.toml --device cuda
```

For the historical fixed-profile comparison, keep the split seed at 17 and vary only the optimisation seed. The command checkpoints every epoch and can resume after an interruption:

```powershell
.venv-gpu\Scripts\transformer-lab.exe run-sentiment-matrix --config configs\sentiment\baseline.toml --config configs\sentiment\head-only-gpt2-small.toml --config configs\sentiment\lora-gpt2-small.toml --config configs\sentiment\full-gpt2-small.toml --seeds 17 23 41 --device cuda --output reports\results\financial-phrasebank-comparison.json --resume
```

The exact commands for the exploratory v3 candidate, selection and final-evaluation stages are in [`docs/gpu-setup.md`](docs/gpu-setup.md). Candidate training and final evaluation write result files and may overwrite their explicitly supplied output paths; use `--resume` only with matching evidence. The read-only replay and verification commands do not write to the tracked result paths.

## Local playground

After GPT-2 parity and the sentiment runs have produced their ignored local artefacts:

```powershell
.venv\Scripts\python.exe -m streamlit run app\playground.py
```

The Generation tab samples from the local implementation loaded with mapped GPT-2 small weights. The Sentiment tab defaults to the validation-selected exploratory v3 profile and also exposes the historical fixed profile; each profile keeps its own verified result records and model paths. Both return genuine model outputs. The app performs no network download while serving a request; missing artefacts produce setup instructions instead.

## What is implemented

- Pre-normalised causal self-attention with learned token and positional embeddings.
- Tied language-model input and output weights.
- An educational byte-level BPE with deterministic merge ordering and UTF-8 round trips.
- A GPT-2 state-dict converter that handles the transposed projection weights used by Hugging Face's Conv1D layers.
- Deterministic batches, separate training/evaluation random streams, gradient accumulation, clipping, optional mixed precision, and atomic checkpoints.
- LoRA injection for selected linear projections, with frozen base weights, adapter-only serialization, and parameter accounting.
- Grouped, stratified Financial PhraseBank splitting that prevents exact duplicates crossing partitions.
- Validation-set temperature scaling with test-set Brier score and expected calibration error.
- Three-seed summaries with aligned paired-bootstrap comparisons.
- Source-tree, configuration and checkpoint fingerprints in new result records.
- Validation-only candidate selection with a frozen selection manifest; test metrics are read only after selection.

The implementation is intended for inspection and reproducible small runs. It is not a claim that a laptop-scale experiment matches a production training system.

## Design decisions

- Raw PyTorch keeps tensor shapes and weight tying visible. Hugging Face supplies the pinned GPT-2 reference and tokenizer only.
- The PhraseBank splitter groups normalised duplicate sentences before stratification. A row-wise split would allow repeated sentences to cross the test boundary.
- Full tuning is the accuracy reference; LoRA makes the parameter/accuracy trade-off measurable rather than implied.
- A fixed split seed and separate optimisation seeds measure training variation without changing the benchmark.
- The scratch transformer uses a byte vocabulary and should not be compared with GPT-2 perplexities without accounting for tokenisation.

## Limits and next steps

The three-seed matrix is stronger than a single run, but it is still one dataset, one GPT-2 scale and one maximum sequence length. The labels are sentence-level judgements from the original annotator pool, not current market impact. The next useful extension is a time-based financial-news holdout, followed by a larger-model comparison if the hardware budget allows it.

See the [derivations](reports/derivations.md), [generated historical table](reports/generated-results.md), [generated v3 table](reports/generated-results-v3.md), [historical figures](reports/figures), [v3 figures](reports/figures/v3), and the [interview walkthrough](docs/walkthrough.md) for the tensor-level explanation and the limits of the evidence. Historical results are fixed-profile evidence; the validation-selected v3 follow-up is labelled exploratory rather than treated as an independent holdout.

## Repository map

```text
src/transformer_lab/   model, data, training, evaluation, CLI, and inference code
configs/               versioned experiment settings
scripts/               thin command-line wrappers and hardware checks
tests/                 unit and integration coverage
reports/               derivations, decisions, and tracked result summaries
docs/                  data/model cards, GPU setup, and release checklist
app/                   local Streamlit playground
```

## Data and model attribution

Financial PhraseBank is distributed by Malo, K. et al. under its published terms; the project downloads it at runtime and does not redistribute the archive. GPT-2 weights and tokenizer files are provided by the `openai-community/gpt2` model repository under its own terms. See [`docs/data-card.md`](docs/data-card.md) and [`docs/model-card.md`](docs/model-card.md) for the exact revisions and limitations.

The source code is licensed under the MIT License.
