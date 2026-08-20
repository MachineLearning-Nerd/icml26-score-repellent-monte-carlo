# ICML 2026 — Score-Repellent Monte Carlo

Paper-level result: INCONCLUSIVE.

This repository is an independent clean-room audit of [Score-Repellent Monte
Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General
State Spaces](https://arxiv.org/abs/2604.22948). It implements NumPy
algorithmic paths, finite diagnostics, continuous-target comparisons, and a
small clean-room Static-MNIST stretch. It is not the authors' official
implementation and does not prove the stochastic-approximation theorems.

The local audit has 5/6 checks passing, one mixed result, and one honest
negative. Zero of six complete paper claims are independently verified. The
repository can be published as an evidence-bound audit, not as a complete
paper reproduction, and no current external score is claimed.

## Paper

- Title: Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces
- Authors: Jie Hu, Lingyun Chen, Geeho Kim, Jinyoung Choi, Bohyung Han, and Do Young Eun
- Paper: [arXiv:2604.22948](https://arxiv.org/abs/2604.22948)
- HTML: [arXiv HTML](https://arxiv.org/html/2604.22948)
- OpenReview: [PN8EiOzMuT](https://openreview.net/forum?id=PN8EiOzMuT)
- Official author repository: [srmc-project/Score-Repellent-Monte-Carlo](https://github.com/srmc-project/Score-Repellent-Monte-Carlo)
- Version pinned: arXiv v2, revised 2026-05-22
- Publication record: accepted at ICML 2026 (Spotlight), as recorded by the paper source

SRMC stores trajectory history as a running average of score evaluations in
R^d. It uses that vector in an exponential score tilt to define a
normalization-free surrogate target and wraps base kernels such as MALA, HMC,
and discrete GWG.

## Claim ledger

| ID | Paper target | Local evidence | Status |
| --- | --- | --- | --- |
| C0 | Constant-memory score history versus an empirical measure | Counts one d-dimensional history vector against the 2^784 discrete state space | FINITE_ACCOUNTING |
| C1 | Proposition 3.4 and Eq. 16 covariance scaling | Exact Gaussian algebra plus one finite SR-MALA sweep | PARTIAL_REPRODUCTION |
| C2 | Theorem 3.3 convergence and joint CLT | 6,000-step, 200-replicate Gaussian diagnostics with convergence and normality checks | PROXY_PASS only |
| C3 | Continuous-target MSE improvement | Gaussian MALA/HMC improvements but both tested logistic cells worsen | MIXED_PARTIAL |
| C4 | Static-MNIST SR-GWG quality improvement | Clean-room RBM and 100-chain stretch gives about 0.8% KL reduction, not 84% | NOT_REPRODUCED |
| C5 | Proposition 3.6 discrete Stein identity | Exhaustive enumeration of an 8-dimensional Ising target | FINITE_EXACT_CHECK |

These statuses describe the local evidence, not complete paper reproduction.
The machine-readable contract is in claims.json, and the detailed
production paths are in CLAIM_EVIDENCE.md.

## How each claim is produced

Core checks:

~~~bash
python3 repro/src/verify.py --claims 0,1,2,3,5
~~~

Static-MNIST stretch:

~~~bash
python3 repro/src/c4_mnist.py --steps 3000 --alpha 1e-4 --epochs 18
python3 repro/src/finalize_gate.py
~~~

1. C0 counts one 784-dimensional history vector against an empirical measure
   over {0,1}^784. This illustrates representation size, not actual runtime
   or total kernel memory.
2. C1 evaluates the Gaussian alpha-dependent block analytically and estimates
   covariance from local SR-MALA runs. The checked analytic tail slope is
   about -0.92 and the local alpha=0-to-20 reduction is about 19.8x.
3. C2 runs finite Gaussian trajectories, checks that the score history
   becomes small, checks the running-mean behavior, and applies Shapiro-Wilk,
   skewness, and kurtosis diagnostics. These are not theorem proofs.
4. C3 compares baseline alpha=0 with SR-MALA and SR-HMC on a correlated
   Gaussian and synthetic Bayesian logistic regression. Gaussian cells
   improve locally, while both tested logistic cells worsen; the result is
   intentionally MIXED_PARTIAL.
5. C4 trains a small clean-room RBM and compares 100-chain GWG and SR-GWG.
   The committed result is about 0.8% KL reduction and Vendi 1.06 to 1.10,
   versus the paper's 84% and 2.6 to 6.4. This is preserved as an honest
   negative.
6. C5 exhaustively enumerates all 2^8 states of a finite Ising target and
   records max absolute E_pi[s_i] of about 8.4e-17. This is an exact finite
   identity check, not a proof of the general proposition.

The final repository verifier checks metadata and publication boundaries
without rerunning the long diagnostics:

~~~bash
python3 verify_final.py
~~~

## Current evidence boundary

- Local checks passing: 5/6.
- C3: mixed across the checked Gaussian and logistic targets.
- C4: honest negative against the paper-scale Static-MNIST result.
- Scoped evidence points: 8/12.
- Complete paper claims independently verified: 0/6.
- Current external score claim: false.
- Author endorsement: not claimed.

The likely contributors to the C4 mismatch include the clean-room RBM versus
the paper's deep EBM, the shorter run, and a relaxed score whose scale makes
alpha=10^-4 nearly inactive locally. These are hypotheses to investigate, not
proof that any single factor is causal.

## Repository map

- repro/src/srmc.py — continuous and discrete SRMC primitives.
- repro/src/verify.py — C0, C1, C2, C3, and C5 local checks.
- repro/src/c4_mnist.py — C4 clean-room Static-MNIST stretch.
- repro/src/finalize_gate.py — canonical verdict and gate report generator.
- outputs/verdict.json — structured raw evidence and claim statuses.
- outputs/verdict_c23.json — retained C2/C3 run evidence.
- outputs/c4_mnist.json — retained C4 negative result.
- outputs/mnist_binary.npz — input data used by the C4 stretch.
- outputs/*.log — retained run logs.
- CLAIM_EVIDENCE.md — claim-to-code-to-output production ledger.
- SOURCE_AUDIT.md — paper provenance and evidence boundary.
- ENVIRONMENT.md — runtime assumptions and commands.
- REPORT.md — publication boundary and unblockers.
- verify_final.py — fail-closed repository-state verifier.

## Branches

The repository publishes one canonical main branch. No experiment, orx/*, or
paper-version branches are in scope. The branch and history policy is in
BRANCH_AUDIT.md.

## Citation

~~~bibtex
@article{hu2026score,
  title={Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces},
  author={Hu, Jie and Chen, Lingyun and Kim, Geeho and Choi, Jinyoung and Han, Bohyung and Eun, Do Young},
  journal={arXiv preprint arXiv:2604.22948},
  year={2026},
  note={Version 2, revised 2026-05-22}
}
~~~

Please cite the paper for its research claims and this repository for the
independent audit artifacts.

## Thank you

Thank you to Jie Hu, Lingyun Chen, Geeho Kim, Jinyoung Choi, Bohyung Han, and
Do Young Eun for developing SRMC and making the algorithmic ideas,
assumptions, and experimental targets clear enough to audit. The
constant-memory score-history perspective made a useful clean-room
implementation possible, and the paper's explicit discussion of the relaxed
discrete score helped us report the Static-MNIST mismatch honestly.

This note is an expression of appreciation only; the authors did not review,
endorse, or maintain this independent repository.

## Attribution

Repository documentation and commits are attributed to
MachineLearning-Nerd using
MachineLearning-Nerd@users.noreply.github.com.
