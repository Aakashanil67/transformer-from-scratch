## Results

| Method | Test macro-F1 (mean ± SD) | Accuracy (mean ± SD) | Peak allocated memory (GiB) | Status |
| --- | ---: | ---: | ---: | --- |
| GPT-2 full fine-tuning | 0.8884 ± 0.0222 | 0.9144 ± 0.0123 | 2.354 GiB | completed |
| GPT-2 head-only | 0.7256 ± 0.0346 | 0.7973 ± 0.0362 | 0.958 GiB | completed |
| GPT-2 LoRA | 0.8958 ± 0.0026 | 0.9234 ± 0.0040 | 0.962 GiB | completed |
| TF-IDF + balanced logistic regression | 0.8155 (single seed) | 0.8687 (single seed) | n/a | completed |
