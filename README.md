# Transformer from scratch

This repository implements a decoder-only transformer in PyTorch and checks it against GPT-2. It includes a byte-level BPE tokenizer, Financial PhraseBank experiments, low-rank adapters, and a local Streamlit playground.

The decoder is implemented from scratch. A small byte-level model learns from random initialisation. A separate model loads pinned pretrained GPT-2 weights. The parity and sentiment results use the converted GPT-2 model.

The model and training code are in `src/transformer_lab`.

## The problem

Can the local transformer reproduce a public GPT-2 checkpoint closely enough to generate text and classify sentences? A falling training loss does not establish that. The forward pass is checked against pinned GPT-2 weights. Sentiment experiments compare adapters with full tuning and a TF-IDF reference. Saved artefacts allow the scores to be checked later.

## Verified historical results (fixed profile)

| Check | Result |
| --- | --- |
| GPT-2 small conversion | 5 fixed prompts. Max absolute logit error `1.125e-4` at a `2e-4` CPU tolerance. Next-token agreement 100% |
| Expanded GPT-2 parity | 4 random token lengths, one mixed-length batch and all hidden-state boundaries. Maximum hidden-state error `4.883e-4` at a `1e-3` tolerance |
| Financial PhraseBank TF-IDF reference | Test macro-F1 `0.8007` (95% bootstrap interval `0.7547–0.8414`). Accuracy `0.8533` |
| Financial PhraseBank GPT-2 head-only | Three-seed test macro-F1 `0.4136 ± 0.0227`. Accuracy `0.6918 ± 0.0150` |
| Financial PhraseBank GPT-2 LoRA | Three-seed test macro-F1 `0.6931 ± 0.0471`. Accuracy `0.7960 ± 0.0175` |
| Financial PhraseBank GPT-2 full tuning | Three-seed test macro-F1 `0.8687 ± 0.0037`. Accuracy `0.8996 ± 0.0039` |
| Tiny Shakespeare byte-bigram | Validation loss `5.990` → `2.772` over 100 updates |
| Tiny Shakespeare scratch transformer | Validation loss `5.558` → `2.162` over 1,000 optimiser updates. The model has 842,496 parameters |

The result summaries are tracked in [`reports/results`](reports/results). Raw data, pretrained weights, checkpoints, and transient run files are ignored. The GPT-2 parity summary uses the pinned `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The Financial PhraseBank archive uses dataset revision `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`.

## Exploratory v3 follow-up (validation-selected)

The current comparison table is [`reports/generated-results-v3.md`](reports/generated-results-v3.md). Validation runs selected the settings, which were frozen in the [selection manifest](reports/results/v3/selection.json) before evaluating seeds 17, 23, and 41. The manifest, [12 candidate summaries](reports/results/v3/candidates), and [per-seed records](reports/results/v3/runs) are retained with the table. This follow-up reuses the historical test partition. Treat it as exploratory, not as an independent holdout or a general ranking of methods.

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

The commands for v3 candidate training, selection, and final evaluation are in [`docs/gpu-setup.md`](docs/gpu-setup.md). They write result files. Use `--resume` only with matching evidence. The same guide lists tests and table checks that read the tracked records without replacing them.

## Local playground

After GPT-2 parity and the sentiment runs have produced their ignored local artefacts:

```powershell
.venv\Scripts\python.exe -m streamlit run app\playground.py
```

The Generation tab samples from the local model with mapped GPT-2 small weights. The Sentiment tab defaults to the exploratory v3 profile and also offers the historical profile. Each uses its own verified records and model paths. The app loads artefacts locally. If any are missing, it shows setup instructions.

## What is implemented

The decoder uses pre-normalised causal self-attention, learned token and positional embeddings, and tied input and output weights. The byte-level BPE learns merges in a fixed order and preserves UTF-8 round trips. The GPT-2 converter transposes the projection weights stored by Hugging Face's Conv1D layers.

Training uses deterministic batches and separate random streams for training and evaluation. It supports gradient accumulation, clipping, mixed precision, and atomic checkpoints. LoRA can replace selected linear projections while freezing the base weights. Adapter weights can be saved separately, and parameter counts are reported.

Financial PhraseBank is split by sentence group so exact duplicates cannot cross partitions. Validation data selects candidates and fits a temperature for probability calibration. Test metrics are read after selection and include Brier score and expected calibration error. Three-seed summaries compare predictions on the same test rows using paired bootstrap resampling. New result records include fingerprints of the source, configuration, and checkpoints.

## Design decisions

- Raw PyTorch keeps tensor shapes and weight tying visible. Hugging Face supplies the pinned GPT-2 reference and tokenizer only.
- The PhraseBank splitter groups normalised duplicate sentences before stratification. A row-wise split would allow repeated sentences to cross the test boundary.
- Full tuning is the accuracy reference. LoRA measures the trade-off between parameter count and accuracy.
- A fixed split seed and separate optimisation seeds measure training variation without changing the benchmark.
- The scratch transformer uses a byte vocabulary and should not be compared with GPT-2 perplexities without accounting for tokenisation.

## Limits and next steps

The three-seed matrix is stronger than a single run, but it is still one dataset, one GPT-2 scale and one maximum sequence length. The labels are sentence-level judgements from the original annotator pool, not current market impact. The next useful extension is a time-based financial-news holdout, followed by a larger-model comparison if the hardware budget allows it.

The [derivations](reports/derivations.md) explain the tensors. The [generated historical table](reports/generated-results.md), [generated v3 table](reports/generated-results-v3.md), [historical figures](reports/figures), and [v3 figures](reports/figures/v3) show the recorded results. The [interview walkthrough](docs/walkthrough.md) follows the main implementation choices and one failed method.

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

Financial PhraseBank is distributed by Malo, K. et al. under its published terms. The project downloads it at runtime and does not redistribute the archive. GPT-2 weights and tokenizer files are provided by the `openai-community/gpt2` model repository under its own terms. See [`docs/data-card.md`](docs/data-card.md) and [`docs/model-card.md`](docs/model-card.md) for the exact revisions and limitations.

The source code is licensed under the MIT License.
