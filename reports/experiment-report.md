# Financial text experiment report

## Question

How much adaptation does GPT-2 small need to compete with a sparse lexical reference on Financial PhraseBank? The experiment compares a frozen representation, rank-4 attention adapters, full fine-tuning, and TF-IDF without allowing exact duplicate sentences to cross partitions.

## Method

The 75-agreement subset is loaded from the pinned dataset revision and grouped by a Unicode-normalised sentence hash. Seed 17 produces 2,417 training rows, 518 validation rows, and 518 test rows. The combined split fingerprint is `f9431e8d61a365a385c865796967754eb4da00e8e05cc3bd94a1c03f53727cf2`. Validation macro-F1 selects the saved transformer epoch; the test partition is evaluated afterwards from a frozen local artefact.

The reference uses word 1–2 gram TF-IDF features (`min_df=2`, sublinear term frequency) and balanced logistic regression. `head_only` trains the 2,307-parameter classifier head. `lora` also trains rank-4 adapters on the attention input and output projections, for a total of 223,491 trainable parameters out of 124,663,299. `full` trains all 124,442,115 parameters. Each transformer run uses GPT-2 small, sequences of at most 64 tokens, micro-batches of one, gradient accumulation over 16 batches, mixed precision, and three epochs. These are historical fixed-profile runs; the three transformer rows report across-seed means and standard deviations, while TF-IDF is a single deterministic seed-17 fit.
Peak allocated memory is the maximum recorded allocation across the seeds for each transformer method, expressed in binary GiB.

## Results

| Method | Test macro-F1 (mean ± SD) | Accuracy (mean ± SD) | Peak allocated VRAM (GiB) | Status |
| --- | ---: | ---: | ---: | --- |
| TF-IDF + balanced logistic regression | 0.8007 (single seed 17) | 0.8533 (single seed 17) | — | completed |
| GPT-2 head-only | 0.4136 ± 0.0227 | 0.6918 ± 0.0150 | 0.958 GiB | completed |
| GPT-2 LoRA | 0.6931 ± 0.0471 | 0.7960 ± 0.0175 | 1.427 GiB | completed |
| GPT-2 full fine-tuning | 0.8687 ± 0.0037 | 0.8996 ± 0.0039 | 3.748 GiB | completed |

For the seed-17 runs, the full model's per-class F1 scores are 0.8348 for negative, 0.9337 for neutral, and 0.8249 for positive. LoRA reaches 0.6542, 0.8908, and 0.6094 respectively. The head-only model never predicts the negative class correctly; its negative-class F1 is zero. Its 0.6795 accuracy obscures that failure because neutral examples form most of the test set. Per-class values here are single-seed diagnostics, not three-seed averages.

## Scratch language-model check

The separate byte-level decoder was initialised randomly and trained on the prepared Tiny Shakespeare split. The four-layer, four-head, width-128 model reached validation loss 2.162 from 5.558 over 1,000 optimiser updates, processing 4,096,000 tokens. This is a learning check for the local implementation, not a perplexity comparison with GPT-2's different tokenizer. The saved result is [`tiny-shakespeare-transformer-scratch.json`](results/tiny-shakespeare-transformer-scratch.json); its ignored local checkpoint is bound by the recorded hash.

## Generated diagnostics

The [per-seed parameter/accuracy plot](figures/macro-f1-vs-parameters.svg), [calibration reliability plot](figures/calibration-reliability.svg), and [scratch learning curve](figures/scratch-transformer-learning-curve.svg) are generated from the saved records and local artefacts. Calibration points show the seed-17 LoRA test scores before and after the validation-fitted temperature; bin counts are printed beside each point.

## Validation-selected follow-up status

The v3 candidate protocol is frozen in [`configs/sentiment/v3-protocol.toml`](../configs/sentiment/v3-protocol.toml). The protocol contains 12 entries—three settings for each of head-only, LoRA, full fine-tuning, and TF-IDF; the original plan shorthand called these nine candidates. All 12 completed successfully, and their validation-only summaries are retained in [`reports/results/v3/candidates`](results/v3/candidates). The selector rejects test-labelled fields and ranks validation macro-F1, validation cross-entropy loss, then protocol order.

The frozen validation choices are head-only `head-only-lr-1e-2`, LoRA `lora-r4-lr-1e-3`, full fine-tuning `full-lr-2e-5`, and TF-IDF `tfidf-c-10`, recorded in the [selection manifest](results/v3/selection.json). The final three-seed matrix used seeds 17, 23, and 41 for each transformer choice plus the deterministic seed-17 TF-IDF reference. Its checked aggregate table is [`generated-results-v3.md`](generated-results-v3.md), with the underlying replayable records in [`reports/results/v3`](results/v3) and derived v3 figures in [`figures/v3`](figures/v3).

This is an exploratory validation-selected follow-up, not an independent holdout: the same fixed test partition is evaluated only after selection, and the selection manifest declares `test_access=false`. The aggregate builder replays the saved test predictions, checks exact example alignment and declared artefact hashes, and then recomputes the displayed means and paired bootstrap comparisons. These results should therefore be read as a post-selection follow-up alongside—not as a replacement for—the historical fixed-profile table above.

## Interpretation

Full fine-tuning produced the strongest result on every seed. The paired bootstrap differences against TF-IDF were positive for all three full-tuning runs (0.06, 0.07 and 0.07 macro-F1; each 95% interval excluded zero). LoRA improved on a frozen representation but remained below TF-IDF on all three aligned comparisons. It trained 0.1793% of the combined model parameters and used substantially less memory than full tuning. This is a result for the fixed split and hardware profile, not a general ranking of adaptation methods.

## Limitations

The dataset is small, and exact grouping does not remove semantic overlap. The three seeds give a first estimate of training variation, not a definitive uncertainty interval. The probability scores are temperature-scaled on validation data but remain model scores, not decision probabilities. The study covers one base-model scale and one maximum sequence length. Training time and memory reflect a particular Windows laptop and software build. The bootstrap intervals describe resampling uncertainty on the fixed test partition; the reported across-seed standard deviations describe training variation. Paired bootstrap comparisons use the same test rows for each method.
