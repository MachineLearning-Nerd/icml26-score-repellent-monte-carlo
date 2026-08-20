# Source audit

## Paper identity

- Title: Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces
- Authors: Jie Hu; Lingyun Chen; Geeho Kim; Jinyoung Choi; Bohyung Han; Do Young Eun
- arXiv: 2604.22948
- OpenReview: PN8EiOzMuT
- Pinned version: v2, revised 2026-05-22
- Paper: https://arxiv.org/abs/2604.22948
- HTML: https://arxiv.org/html/2604.22948
- Author repository: https://github.com/srmc-project/Score-Repellent-Monte-Carlo

## Available source

The repository includes the NumPy SRMC primitives, finite Gaussian and
discrete checks, a clean-room Static-MNIST stretch, retained logs and raw
outputs, and a finalizer that merges the evidence. The author repository is
linked for provenance; this audit does not claim to have imported or run it.

## Evidence boundary

The local evidence spans finite representation accounting, Gaussian
algebra, finite trajectories, local MSE comparisons, a small RBM experiment,
and exhaustive enumeration of one finite Ising target. It omits the full
stochastic-approximation proofs, general-target guarantees, paper-scale
continuous benchmark curves, author EBM, and exact paper training setup.

C3 is mixed rather than a universal pass. C4 is an honest negative against
the reported Static-MNIST numbers, with plausible configuration differences
documented as hypotheses rather than causal conclusions.

## Audit conclusion

The six-claim contract is pinned. Five local checks pass, C3 is mixed, and C4
is not reproduced. No complete paper-level claim is independently verified.
