# Financial PhraseBank data card

## Source

The experiment downloads the `financial_phrasebank` dataset from [Hugging Face](https://huggingface.co/datasets/takala/financial_phrasebank) at revision `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`. It reads the `Sentences_75Agree.txt` member of `FinancialPhraseBank-v1.0.zip` and applies the archive's Latin-1 decoding.

The dataset contains financial news sentences labelled `negative`, `neutral`, or `positive`. The project uses the 75-agreement subset, with 3,453 rows and 3,448 exact sentence groups. Five rows are repeated sentences; no duplicate group crosses a split.

## Split and provenance

The seeded grouped split uses seed 17 and allocates 2,417 training rows, 518 validation rows, and 518 test rows. The archive SHA-256 is `0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6`.

Each tracked result stores partition sizes, class counts, per-partition SHA-256 fingerprints, and the combined split fingerprint `f9431e8d61a365a385c865796967754eb4da00e8e05cc3bd94a1c03f53727cf2`. It does not contain the original sentences or the full row-level manifest. The raw archive and local manifest stay in ignored experiment storage and are not redistributed by this repository.

## Intended use

The split supports a small comparison between a TF-IDF reference classifier and GPT-2-based classification modes. Macro-F1 is the primary metric because the labels are imbalanced. Accuracy and per-class scores are reported as secondary measures.

## Limitations

The labels are sentence-level financial sentiment judgements, not investment advice or a measure of market impact. Exact deduplication prevents one obvious leakage path, but semantically related news may still occur across partitions. The dataset is small and the experiments do not establish performance on current financial reporting or other languages.

The dataset card identifies the source licence as Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported. This project downloads the archive at runtime and does not redistribute it. Users must review the [dataset card](https://huggingface.co/datasets/takala/financial_phrasebank) and licence before redistributing the data or derived material. Cite Malo, Sinha, Korhonen, Wallenius and Takala (2014), “Good debt or bad debt: Detecting semantic orientations in economic texts”, *Journal of the Association for Information Science and Technology*, 65(4), 782–796, [doi:10.1002/asi.23062](https://doi.org/10.1002/asi.23062).
