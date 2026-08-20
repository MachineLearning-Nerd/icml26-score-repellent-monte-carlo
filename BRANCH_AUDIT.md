# Branch and commit audit

## Normalization

- Source tip before standardization:
  f3256f6bc984577564a30462fc43a99f6fe2194e.
- The source and live repository publish one branch, main.
- No experiment, orx/*, or paper-version branch is in scope.
- Recovery bundle SHA-256:
  be22286b8b090e2b6120f03593138cd4b131399526def6f88737fa2e99e37e98.
- The recovery bundle preserves the complete pre-normalization history.

## Current branch policy

- main is the canonical paper metadata, evidence, and verifier branch.
- New evidence should use a descriptive audit/... branch and record the paper
  version, command, inputs, outputs, controls, and limitations before merging.
- C4's negative result must remain visible; it must not be silently replaced
  by a pass without new paper-faithful evidence.

## Attribution policy

All reachable commits in the published history use:

MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>

The earlier local loop-srmc identity and co-author trailer are retained only
in the recovery bundle and history notes, not in live commit metadata.
