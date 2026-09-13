# Project decisions

The model code stays in raw PyTorch so that attention, residual paths, weight tying, and adapter updates remain visible. Hugging Face is used only as a pinned reference for GPT-2 weights and its tokenizer contract.

Macro-F1 is the primary sentiment metric. The dataset is split by exact sentence group, with validation used for choices and the test set reserved for the final report. The TF-IDF reference and all three transformer modes use the same frozen split.

Result summaries are small JSON files under `reports/results`. Raw data, downloaded weights, checkpoints, and local run directories stay outside Git. A result can be `completed`, `failed`, or `unavailable`; a failed or unavailable run is not converted into a score.

The laptop's GTX 1650 runs the GPT-2 profiles in an isolated CUDA 12.8 environment, leaving the CPU test environment intact. Each transformer result records CUDA availability, package versions, peak allocation, and peak reservation.

The manual attention implementation keeps query/key/value shapes and the causal mask inspectable; a fused kernel would be a useful later performance comparison, but would obscure the first-pass derivation. The byte tokenizer is used for the from-scratch language-model demonstration because every UTF-8 byte is representable; the converted GPT-2 path retains GPT-2's own tokenizer so parity measures the model rather than a tokenisation change. Sentiment uses a fixed maximum length of 64, with truncation reported descriptively rather than tuned on the test partition.

The sentiment study freezes the backbone for head-only training, injects rank-4 LoRA into attention input/output projections, and unfreezes all parameters for full tuning. These choices expose the parameter/accuracy/memory trade-off. The historical learning rates are method profiles, not a claim that each method received an exhaustive search; the validation-selected follow-up records its candidate budget separately.
