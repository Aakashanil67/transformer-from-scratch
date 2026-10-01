# Release checklist

Use this checklist to record release checks. Mark an item only after the command passes, and record the environment used.

## Reproducibility

- [ ] CPU wheel install and outside-checkout CLI workflow.
- [ ] Pinned GPT-2 parity with temporary output paths.
- [ ] Frozen Financial PhraseBank split and dataset hash replay.
- [ ] Every displayed metric maps to a verified result JSON and artefact hash.

## Quality

- [ ] Full CPU suite, coverage floor, formatting and lint.
- [ ] CPU integration contracts and conditional CUDA contracts.
- [ ] Report/table/figure consistency check.
- [ ] App success and failure-state checks. Browser interaction is identified separately from health checks.

## Publication

- [ ] Private plans, raw data, checkpoints, caches and logs remain ignored.
- [ ] Dataset and model terms are acknowledged separately from the MIT code licence.
- [ ] Staged filenames, secrets, schemas and the complete diff are reviewed.
- [ ] Review the final diff and commit message after the checks pass.
