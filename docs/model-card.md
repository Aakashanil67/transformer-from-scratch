# Model card

## Local decoder

The repository implements a decoder-only transformer with learned token and position embeddings, pre-normalised residual blocks, causal multi-head attention, GELU feed-forward layers, and tied input/output embeddings. The educational byte-level BPE is separate from the GPT-2 tokenizer.

The GPT-2 conversion targets `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The local loader transposes the four projection matrices whose upstream Conv1D layout differs from PyTorch's `nn.Linear` layout.

The scratch-training path is a separate byte-level decoder initialised from random weights. Its learning curves and samples must not be presented as converted GPT-2 generations. The two paths answer different questions.

The five-prompt CPU parity run reached a maximum absolute logit error of `1.125e-4` at a tolerance of `2e-4`, with 100% next-token agreement. The expanded suite adds four random token lengths, a mixed-length batch and hidden-state comparisons. Its maximum hidden-state error is `4.883e-4` at a tolerance of `1e-3`. These checks validate weight mapping and forward computation. They do not measure generated-text quality, safety or domain accuracy.

## Sentiment head

The sentiment model pools the last non-padding hidden state and applies a three-class linear head. The supported modes are `head_only`, `lora`, and `full`. LoRA targets the attention `c_attn` and `c_proj` projections by default. The base weights remain frozen.

The recorded GTX 1650 runs reached three-seed test macro-F1 means of 0.4136 (SD 0.0227) for head-only training, 0.6931 (SD 0.0471) for LoRA, and 0.8687 (SD 0.0037) for full fine-tuning. The TF-IDF reference reached 0.8007 on the same frozen 75-agreement split. These are measurements on a small benchmark, not estimates of performance on live financial news. The transformer probabilities use one scalar temperature fitted on validation data. Result records also report Brier score and expected calibration error. Calibration does not make the model suitable for investment decisions.

The exploratory v3 profile selects settings from validation-only candidate runs and then evaluates the same three optimisation seeds (17, 23 and 41). Its observed test macro-F1 means are 0.7256 (SD 0.0346) for head-only, 0.8958 (SD 0.0026) for LoRA, and 0.8884 (SD 0.0222) for full tuning. The TF-IDF reference is 0.8155 from one deterministic fit. The paired bootstrap intervals for full tuning minus LoRA all include zero. Because this follow-up reuses the inspected test partition, it is exploratory and does not establish a general advantage.

## Intended use and limitations

This is an educational and portfolio implementation. It supports inspection of tensor shapes, loading public checkpoints, studying low-rank updates, and repeating small local experiments. It is not a production serving stack, an investment system, or evidence that LoRA outperforms full fine-tuning on financial text. The historical low head-only score is a result for fixed optimisation settings. The v3 candidate with a higher learning rate improved the measured score without changing the frozen representation, so it should not be interpreted as proof of an architectural ceiling.

The upstream model and tokenizer terms remain applicable to the downloaded files. Review those terms before redistribution.
