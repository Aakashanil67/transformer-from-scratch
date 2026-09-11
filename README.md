# Transformer from scratch

This repository implements a compact decoder-only transformer in raw PyTorch and then checks its computations against GPT-2. It also provides a byte-level BPE for study, a reproducible Financial PhraseBank reference experiment, low-rank adapter components, and a local Streamlit playground.

The project is deliberately inspectable. The important operations are in `src/transformer_lab`, rather than hidden behind a training framework.

## Verified results

| Check | Result |
| --- | --- |
| GPT-2 small conversion | 5 fixed prompts; max absolute logit error `1.125e-4` at a `2e-4` CPU tolerance; next-token agreement 100% |
| Financial PhraseBank TF-IDF reference | Test macro-F1 `0.8007` (95% bootstrap interval `0.7547–0.8414`); accuracy `0.8533` |
| Financial PhraseBank GPT-2 head-only | Test macro-F1 `0.3945`; accuracy `0.6795`; 2,307 trainable parameters |
| Financial PhraseBank GPT-2 LoRA | Test macro-F1 `0.7182` (95% interval `0.6666–0.7660`); accuracy `0.8031`; 223,491 trainable parameters |
| Financial PhraseBank GPT-2 full tuning | Test macro-F1 `0.8645` (95% interval `0.8250–0.9005`); accuracy `0.8958` |
| Tiny Shakespeare byte-bigram | Validation loss `5.990` → `2.772` over 100 updates |

The result summaries are tracked in [`reports/results`](reports/results). Raw data, pretrained weights, checkpoints, and transient run files are ignored. The GPT-2 parity summary uses the pinned `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The Financial PhraseBank archive uses dataset revision `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`.

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

Run the Financial PhraseBank reference model:

```powershell
.venv\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\baseline.toml --device cpu
.venv\Scripts\transformer-lab.exe evaluate-sentiment --config configs\sentiment\baseline.toml --device cpu
```

The GPT-2 sentiment profiles share the same duplicate-safe split and metric code. The recorded runs used the documented CUDA 12.8 environment on a GeForce GTX 1650:

```powershell
.venv-gpu\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\head-only-gpt2-small.toml --device cuda
.venv-gpu\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\lora-gpt2-small.toml --device cuda
.venv-gpu\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\full-gpt2-small.toml --device cuda
.venv-gpu\Scripts\transformer-lab.exe evaluate-sentiment --config configs\sentiment\lora-gpt2-small.toml --device cuda
```

## Local playground

After GPT-2 parity and the sentiment runs have produced their ignored local artefacts:

```powershell
.venv\Scripts\python.exe -m streamlit run app\playground.py
```

The Generation tab samples from the local implementation loaded with mapped GPT-2 small weights. The Sentiment tab defaults to the local LoRA checkpoint and retains TF-IDF as a comparison. Both return genuine model outputs. The app performs no network download while serving a request; missing artefacts produce setup instructions instead.

## What is implemented

- Pre-normalised causal self-attention with learned token and positional embeddings.
- Tied language-model input and output weights.
- An educational byte-level BPE with deterministic merge ordering and UTF-8 round trips.
- A GPT-2 state-dict converter that handles the transposed projection weights used by Hugging Face's Conv1D layers.
- Deterministic batches, separate training/evaluation random streams, gradient accumulation, clipping, optional mixed precision, and atomic checkpoints.
- LoRA injection for selected linear projections, with frozen base weights, adapter-only serialization, and parameter accounting.
- Grouped, stratified Financial PhraseBank splitting that prevents exact duplicates crossing partitions.

The implementation is intended for inspection and reproducible small runs. It is not a claim that a laptop-scale experiment matches a production training system.

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
