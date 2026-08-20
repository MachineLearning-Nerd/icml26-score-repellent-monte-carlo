# Publication report

## Current result

This repository is a documented mixed-results audit of SRMC.

- C0: FINITE_ACCOUNTING
- C1: PARTIAL_REPRODUCTION
- C2: PROXY_PASS only
- C3: MIXED_PARTIAL
- C4: NOT_REPRODUCED; honest negative retained
- C5: FINITE_EXACT_CHECK
- Local checks passing: 5/6
- Scoped evidence points: 8/12
- Complete paper-level claims independently verified: 0/6
- Current external score: not claimed
- Author endorsement: not claimed

## Publication boundary

The repository can be published as an evidence-bound audit package. It
cannot be presented as a complete reproduction of SRMC. The
publication_allowed field is false for a complete paper reproduction, even
though the documentation gate is complete.

## Unblockers

The next work should reproduce the author-faithful general-target and
paper-scale continuous experiments, the stochastic-approximation assumptions
and covariance statements, and the deep EBM Static-MNIST setup with matching
data, score scale, initialization, chains, and metrics. Any status upgrade
must preserve the C3 mixed result and the C4 negative control.
