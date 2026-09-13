## Results

| Method | Test macro-F1 (mean ± SD) | Accuracy (mean ± SD) | Peak allocated memory (GiB) | Status |
| --- | ---: | ---: | ---: | --- |
| GPT-2 full fine-tuning | 0.8687 ± 0.0037 | 0.8996 ± 0.0039 | 3.748 GiB | completed |
| GPT-2 head-only | 0.4136 ± 0.0227 | 0.6918 ± 0.0150 | 0.958 GiB | completed |
| GPT-2 LoRA | 0.6931 ± 0.0471 | 0.7960 ± 0.0175 | 1.427 GiB | completed |
| TF-IDF + balanced logistic regression | 0.8007 (single seed) | 0.8533 (single seed) | — | completed |
