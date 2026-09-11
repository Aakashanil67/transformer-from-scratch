# Model card

## Local decoder

The repository implements a decoder-only transformer with learned token and position embeddings, pre-normalised residual blocks, causal multi-head attention, GELU feed-forward layers, and tied input/output embeddings. The educational byte-level BPE is separate from the GPT-2 tokenizer.

The GPT-2 conversion targets `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The local loader transposes the four projection matrices whose upstream Conv1D layout differs from PyTorch's `nn.Linear` layout.

The five-prompt CPU parity run reached a maximum absolute logit error of `1.125e-4` at a tolerance of `2e-4`, with 100% next-token agreement. This validates the weight mapping and forward computation. It does not measure generated-text quality, safety, or domain accuracy.

## Sentiment head

The sentiment model pools the last non-padding hidden state and applies a three-class linear head. The supported modes are `head_only`, `lora`, and `full`. LoRA targets the attention `c_attn` and `c_proj` projections by default; the base weights remain frozen.

The recorded GTX 1650 runs reached test macro-F1 scores of 0.3945 for head-only training, 0.7182 for LoRA, and 0.8645 for full fine-tuning. The TF-IDF reference reached 0.8007. These are single-seed measurements on the frozen 75-agreement split; they are not estimates of performance on live financial news. The class probabilities shown by the demo are uncalibrated softmax outputs.

## Intended use and limitations

This is an educational and portfolio implementation. It supports inspection of tensor shapes, public-checkpoint loading, low-rank update study, and reproduction of small local experiments. It is not a production serving stack, an investment system, or evidence that LoRA outperforms full fine-tuning on financial text.

The upstream model and tokenizer terms remain applicable to the downloaded files. Review those terms before redistribution.
