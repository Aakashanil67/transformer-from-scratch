# Interview walkthrough

This is a short route through the project before discussing implementation details.

## 1. Follow one token through attention

Start with `X` shaped `(B, T)`. Token and position embeddings produce `H` shaped `(B, T, d)`. A single linear layer produces `(Q, K, V)` together, reshaped to `(B, heads, T, d_k)`. The score matrix is `QKᵀ / sqrt(d_k)`. The lower-triangular mask replaces future scores with negative infinity, so a softmax row can only read its own and earlier positions. The heads are concatenated and projected back to width `d`, then added through the pre-normalised residual path.

## 2. Explain the GPT-2 Conv1D mapping

The upstream GPT-2 `Conv1D` projection stores a matrix with the opposite orientation from PyTorch `nn.Linear`. The converter therefore transposes the four projection weights while copying embeddings, LayerNorm parameters and biases directly. A fixed-prompt parity run checks logits and next-token agreement. It does not claim that parity alone proves useful generations.

## 3. Calculate the adapter budget

For width `d` and rank `r`, attention-only LoRA adds `r(d + 3d)` for `c_attn` and `r(d + d)` for `c_proj` per block: `6rd`. Across `L` blocks this is `6Lrd`. At GPT-2 small's `L=12`, `d=768`, `r=4`, that is 221,184 adapter parameters. The classifier head adds 2,307.

## 4. Reproduce one result

Run the CPU tests first, then the pinned parity command. For sentiment, use the recorded seed-17 LoRA artefact only after its result and checkpoint hashes verify. The report's historical table shows a single TF-IDF fit and three-seed transformer means. The per-class paragraph is explicitly a seed-17 diagnostic.

## 5. Explain the strongest failed method

Head-only training is the useful historical failure: the recorded profile with a low learning rate reaches 0.4136 mean test macro-F1 and never predicts the negative class correctly in the seed-17 diagnostic. The v3 validation-selected higher head learning rate improves the measured score without changing the frozen representation, so the older result does not establish a limit of the representation. LoRA improves in the historical profile but remains below TF-IDF there. In the exploratory v3 profile it has the highest observed mean, while the paired intervals do not establish a clear advantage over full tuning. These are fixed-split observations, not universal method rankings.

The scratch language-model command is a third, separate story: it trains the local decoder from random weights on byte tokens. Its loss curve should be read as evidence that the implementation learns, not as a directly comparable GPT-2 perplexity.
