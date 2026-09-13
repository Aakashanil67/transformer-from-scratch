# Release checklist

This checklist is scoped to the verification run recorded with the release commit. A checked item means the named command passed in the named environment; it is not a guarantee of production readiness or of perfect reviewer scores.

## Reproducibility

- [ ] CPU wheel install and outside-checkout CLI workflow.
- [ ] Pinned GPT-2 parity with temporary output paths.
- [ ] Frozen Financial PhraseBank split and dataset hash replay.
- [ ] Every displayed metric maps to a verified result JSON and artefact hash.

## Quality

- [ ] Full CPU suite, coverage floor, formatting and lint.
- [ ] CPU integration contracts and conditional CUDA contracts.
- [ ] Report/table/figure consistency check.
- [ ] App success and failure-state checks; browser interaction is identified separately from health checks.

## Publication

- [ ] Private plans, raw data, checkpoints, caches and logs remain ignored.
- [ ] Dataset and model terms are acknowledged separately from the MIT code licence.
- [ ] Staged filenames, secrets, schemas and the complete diff are reviewed.
- [ ] Exactly one final commit is created after the above checks; no push is part of this workflow.
