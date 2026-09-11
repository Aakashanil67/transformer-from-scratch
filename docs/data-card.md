# Financial PhraseBank data card

## Source

The experiment downloads the `financial_phrasebank` dataset from [Hugging Face](https://huggingface.co/datasets/financial_phrasebank) at revision `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`. It reads the `Sentences_75Agree.txt` member of `FinancialPhraseBank-v1.0.zip` and applies the archive's Latin-1 decoding.

The dataset contains financial news sentences labelled `negative`, `neutral`, or `positive`. The project uses the 75-agreement subset, with 3,453 rows and 3,448 exact sentence groups. Five rows are repeated sentences; no duplicate group crosses a split.

## Split and provenance

The seeded grouped split uses seed 17 and allocates 2,417 training rows, 518 validation rows, and 518 test rows. The archive SHA-256 is `0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6`.

Each tracked result stores partition sizes, class counts, per-partition SHA-256 fingerprints, and the combined split fingerprint `f9431e8d61a365a385c865796967754eb4da00e8e05cc3bd94a1c03f53727cf2`. It does not contain the original sentences or the full row-level manifest. The raw archive and local manifest stay in ignored experiment storage and are not redistributed by this repository.

## Intended use

The split supports a small comparison between a TF-IDF reference classifier and GPT-2-based classification modes. Macro-F1 is the primary metric because the labels are imbalanced. Accuracy and per-class scores are reported as secondary measures.

## Limitations

The labels are sentence-level financial sentiment judgements, not investment advice or a measure of market impact. Exact deduplication prevents one obvious leakage path, but semantically related news may still occur across partitions. The dataset is small and the experiments do not establish performance on current financial reporting or other languages.

Users must review and follow the dataset's own licence and citation terms before redistributing it or derived material.
