# Transformer derivation notes

These notes connect the equations to the tensors in `src/transformer_lab`. They describe identities used by the implementation; empirical statements appear in the experiment report instead.

## Embeddings and positions

Let a batch of token IDs be `X` with shape `(B, T)`. The token embedding matrix `E` has shape `(V, d)`, so `E[X]` has shape `(B, T, d)`. The position matrix `P` has shape `(T_max, d)`. For a sequence of length `T`, the model adds `P[0:T]` to every batch row:

`H_0[b, t, :] = E[X[b, t], :] + P[t, :]`.

The position lookup is independent of the batch. A token therefore receives the same lexical vector at different positions but a different positional vector.

## Scaled causal attention

For one head, the normalised hidden states produce

`Q = HW_Q`, `K = HW_K`, and `V = HW_V`.

With head width `d_k`, the score matrix is

`S = QK^T / sqrt(d_k)`.

The causal mask `M` is zero on and below the diagonal and negative infinity above it. The implementation computes

`A = softmax(S + M)` and `O = AV`.

Consider three positions. Before masking, position 1 could assign probability to keys 1, 2, or 3. After masking, its row is `[s_11, -infinity, -infinity]`, so `softmax` returns `[1, 0, 0]`. Position 2 receives keys 1 and 2; position 3 receives all three. No row can read a future value. The factor `1/sqrt(d_k)` keeps the dot-product scale approximately stable as the head width changes.

The model splits `(B, T, 3d)` into Q, K, and V, reshapes each to `(B, n_head, T, d_k)`, transposes the sequence and head axes for the matrix products, and reverses that operation before the output projection.

## Pre-normalisation and residuals

One transformer block applies

`H' = H + Attention(LN_1(H))`

followed by

`H_next = H' + MLP(LN_2(H'))`.

The residual path carries the previous representation directly to the next block. Layer normalisation acts on the sublayer input rather than after the addition. This is the pre-normalised ordering used by the local model and by GPT-2's converted weights.

For a vector `x` of width `d`, layer normalisation computes

`LN(x)_i = gamma_i (x_i - mu) / sqrt(sigma^2 + epsilon) + beta_i`,

where `mu` and `sigma^2` are the mean and variance over the final dimension. `gamma` and `beta` are learned affine parameters when bias is enabled.

## Cross-entropy and tied output weights

For logits `z` and target class `y`, let `p_i = exp(z_i) / sum_j exp(z_j)`. The token loss is `L = -log(p_y)`. Combining the softmax derivative with the negative log likelihood gives

`dL/dz_i = p_i - 1[i = y]`.

The model passes unnormalised logits directly to PyTorch cross-entropy. Applying softmax before the loss would discard the numerically stable fused implementation.

The language-model head has weight matrix `W` with shape `(V, d)`. The local model assigns the same parameter object to the token embedding and `lm_head.weight`, so the output logit for token `i` is the dot product between the final hidden state and the input embedding row `W_i`. This reduces parameters and ties the input and output token geometry.

## Byte-level BPE

The tokenizer starts with the 256 possible byte values. Given a UTF-8 byte sequence, it counts adjacent pairs and chooses the pair with highest frequency; ties are resolved by the numeric pair. If `(a, b)` is assigned new ID `n`, its vocabulary entry is the concatenation `v[n] = v[a] + v[b]`. Encoding applies learned merges in order. Decoding concatenates the stored byte strings and decodes UTF-8.

For `aaaaaa`, the first frequent pair is `(a, a)`, which becomes one token representing `aa`; the sequence becomes `aaa`. A later rule can merge `(aa, aa)` if the vocabulary budget and corpus frequency allow it. Because the base alphabet contains every byte and the vocabulary stores byte strings, a successful encode/decode round trip is lossless for valid UTF-8. The implementation is intentionally educational: it has no GPT-2 pre-tokenisation rules and recomputes pair counts at each merge.

## LoRA

For a frozen linear projection `W` with input width `d_in` and output width `d_out`, LoRA adds

`W_eff = W + (alpha / r) B A`,

where `A` has shape `(r, d_in)` and `B` has shape `(d_out, r)`. The update has rank at most `r`, and its trainable parameter count is `r(d_in + d_out)` instead of `d_in d_out + d_out` for the base projection.

The local adapter initialises `B` to zero. Therefore the first adapter forward pass equals the frozen base exactly. On the first backward pass, `B` receives a gradient while `A` receives zero because the path through `A` is multiplied by the zero matrix. Once `B` has moved away from zero, both factors receive gradients; the base parameters remain frozen. For GPT-2 small, targeting `c_attn` and `c_proj` in all 12 blocks adds 221,184 adapter parameters. With the 2,307-parameter classifier head, the measured LoRA run trains 223,491 of 124,663,299 parameters.

## GPT-2 conversion

Hugging Face stores GPT-2's projection matrices through Conv1D modules. The local implementation uses `nn.Linear`, whose weight convention is transposed relative to those Conv1D tensors. The loader copies embeddings, layer norms, biases, and projections by an explicit key map and transposes the four projection weights. It checks source presence and target shapes before copying.

The parity experiment compares logits from both models on fixed tokenised prompts. A small maximum error and matching next-token argmax show that the conversion and forward path agree for those inputs. They do not establish that a newly trained local model has useful generations or sentiment accuracy.
