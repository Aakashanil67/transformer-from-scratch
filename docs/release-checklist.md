# Release checklist

## Reproducibility

- [x] Fresh CPU installation completes from the documented commands.
- [x] The pinned GPT-2 parity run passes and its summary is tracked.
- [x] The frozen Financial PhraseBank split and dataset hash are recorded.
- [x] Each published metric has a corresponding result JSON file.

## Quality

- [x] Unit and available integration tests pass; network parity is recorded separately.
- [x] Formatting, lint, coverage, and package build checks pass.
- [x] The command-line workflows perform real work.
- [x] The local demo handles both successful and missing-artefact cases.

## Publication

- [x] Private plans, editor settings, raw data, checkpoints, caches, and logs are ignored.
- [x] Dataset and model licences are acknowledged.
- [x] Clean-room installation, CLI, training, and app smoke checks pass.
- [x] The staged diff contains only intended public files.
