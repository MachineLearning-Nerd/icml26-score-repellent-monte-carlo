# ICML 2026 — Score-Repellent Monte Carlo

Paper-level status: **INCONCLUSIVE — partial evidence, with one honest
negative reproduction**

This repository is an independent clean-room audit of **Score-Repellent Monte
Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General
State Spaces**. It implements the paper's main algorithmic paths in NumPy and
records finite checks and small experiments. It is not the authors' official
implementation and does not prove the stochastic-approximation theorems.

## Paper

- Title: *Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler
  with Constant Memory in General State Spaces*
- Authors: Jie Hu, Lingyun Chen, Geeho Kim, Jinyoung Choi, Bohyung Han, and
  Do Young Eun
- arXiv: [2604.22948](https://arxiv.org/abs/2604.22948)
- Paper HTML: [arXiv HTML](https://arxiv.org/html/2604.22948)
- OpenReview record: [PN8EiOzMuT](https://openreview.net/forum?id=PN8EiOzMuT)
- Official author repository:
  [srmc-project/Score-Repellent-Monte-Carlo](https://github.com/srmc-project/Score-Repellent-Monte-Carlo)
- Pinned claim contract: arXiv **v2**, revised **2026-05-22**
- arXiv record: accepted at **ICML 2026 (Spotlight)**

SRMC stores trajectory history as a running average of score evaluations
theta_n in R^d. It uses that vector in an exponential score tilt to define a
normalization-free surrogate target, then wraps base kernels such as MALA,
HMC, and discrete GWG. The paper analyzes stochastic approximation, a joint
central limit theorem, and regimes where asymptotic covariance decreases as
the repellence strength alpha increases.

## Evidence summary

The local audit has **5/6 checks passing** and **one honest negative**. That is
not the same as verifying five paper claims: the theorem checks are finite
diagnostics, C3 has mixed target-dependent results, and C4 does not match the
paper's Static-MNIST result. No full paper claim is marked as independently
verified.

| ID | Paper claim | How this repository produces evidence | Status |
| --- | --- | --- | --- |
| C0 | Score history uses O(d) memory instead of an empirical measure over the state space | Counts one d-dimensional history vector against 2^d discrete states | FINITE_ACCOUNTING |
| C1 | Proposition 3.4 and Eq. 16 give O(1/alpha) covariance scaling | Exact Gaussian Eq.-15 algebra plus one finite SR-MALA sweep over alpha | PARTIAL_REPRODUCTION |
| C2 | Theorem 3.3 gives almost-sure convergence and a joint CLT | 6,000-step, 200-replicate Gaussian runs, convergence thresholds, Shapiro-Wilk, skewness, and kurtosis | PROXY_PASS |
| C3 | Continuous targets show lower MSE, up to about 5x in the paper | 10-D correlated Gaussian and Bayesian-logistic MALA/HMC local MSE comparisons | MIXED_PARTIAL |
| C4 | Static-MNIST SR-GWG gives 84% KL reduction and Vendi 2.6 to 6.4 | Clean-room RBM plus 100-chain GWG/SR-GWG stretch experiment | NOT_REPRODUCED |
| C5 | Proposition 3.6 gives E_pi[s_i]=0 for the discrete score | Exhaustive enumeration of an 8-D Ising target | FINITE_EXACT_CHECK |

The canonical machine-readable result is
[outputs/verdict.json](outputs/verdict.json). It keeps raw values beside the
status and limitation for each claim.

## Claim-by-claim details

### C0 — constant-memory accounting

repro/src/verify.py counts one 784-dimensional history vector against an
empirical measure over {0,1}^784. This illustrates the representation-size
argument. It is not a runtime or memory benchmark and does not measure the
additional memory used by a real base kernel.

### C1 — covariance scaling

claim_c1_scaling in repro/src/verify.py evaluates the Gaussian alpha-dependent
matrix block analytically and estimates covariance from independent local
SR-MALA runs. The analytic tail slope is about -0.92 and the local
alpha=0-to-20 history covariance reduction is about 19.8x in the checked run.
This supports the Gaussian trend but does not prove Proposition 3.4 for
general targets or kernels.

### C2 — stochastic approximation and CLT

claim_c2_convergence_clt runs finite Gaussian trajectories and checks that the
history becomes small, the running mean is near its target, and selected
normalized coordinates are compatible with a normality diagnostic. These
checks cannot establish almost-sure convergence, all theorem assumptions, or
the joint CLT.

### C3 — continuous-target MSE

claim_c3_mse compares baseline alpha=0 with SR-MALA and SR-HMC on a correlated
Gaussian and synthetic Bayesian logistic regression. In the checked output,
Gaussian MALA improves about 2.18x and Gaussian HMC about 3.10x. The local
logistic-regression ratios are about 0.017 for MALA and 0.000024 for HMC,
meaning the tested SR settings are much worse than baseline there. The paper's
100-replicate step-count and CPU-time curves are therefore not reproduced by
this audit.

### C4 — Static-MNIST mode mixing

repro/src/c4_mnist.py trains a small clean-room RBM and compares 100-chain GWG
and relaxed-score SR-GWG from a digit-7 initialization. The committed result
is approximately 0.8% KL reduction and Vendi 1.06 to 1.10, compared with the
paper's reported 84% KL reduction and Vendi 2.6 to 6.4. The negative result is
preserved rather than converted into a pass.

The likely contributors are the different RBM versus deep EBM, shorter
3,000-step run, and a relaxed score whose scale makes alpha=10^-4 nearly
inactive in this local model. These are explanations to investigate, not
proof that any single factor is causal.

### C5 — discrete Stein identity

claim_c5_discrete_stein exhaustively enumerates all 2^8 states of a finite
Ising target and obtains max absolute E_pi[s_i] of about 8.4e-17. This is an
exact finite check of the identity for that target, not a formal proof of the
general proposition.

## How to run the audit

The core checks use NumPy and SciPy:

~~~bash
python3 repro/src/verify.py --claims 0,1,2,3,5
~~~

The Static-MNIST stretch is separate and can take several minutes:

~~~bash
python3 repro/src/c4_mnist.py --steps 3000 --alpha 1e-4 --epochs 18
python3 repro/src/finalize_gate.py
~~~

The finalizer merges the core verdict with outputs/c4_mnist.json and writes
outputs/verdict.json plus publication_gate.json. The checked-in outputs are
the evidence for the status reported here.

## Repository map

- repro/src/srmc.py — continuous and discrete SRMC algorithmic primitives.
- repro/src/verify.py — C0, C1, C2, C3, and C5 local checks.
- repro/src/c4_mnist.py — C4 clean-room Static-MNIST stretch.
- repro/src/finalize_gate.py — canonical status and gate report generator.
- outputs/verdict.json — structured raw evidence and claim statuses.
- outputs/verdict_c23.json — retained C2/C3 run evidence.
- outputs/c4_mnist.json — retained C4 negative result.
- outputs/mnist_binary.npz — input data used by the C4 stretch.
- outputs/*.log — run logs for the checked evidence.
- STATUS.md — concise handoff.
- BRANCH_AUDIT.md — branch and commit-attribution record.
- GATE_READY.md — publication-scope gate explanation.

## Branches

The repository currently has one canonical branch, main. No experiment,
orx/*, or paper-version branches are in scope. The initial reachable commits
were created by the local loop-srmc identity; the publication cleanup rewrites
reachable attribution to MachineLearning-Nerd.

## Citation

~~~bibtex
@article{hu2026score,
  title={Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces},
  author={Hu, Jie and Chen, Lingyun and Kim, Geeho and Choi, Jinyoung and Han, Bohyung and Eun, Do Young},
  journal={arXiv preprint arXiv:2604.22948},
  year={2026},
  note={Version 2, revised 2026-05-22; accepted at ICML 2026}
}
~~~

## Thank you

Thank you to Jie Hu, Lingyun Chen, Geeho Kim, Jinyoung Choi, Bohyung Han, and
Do Young Eun for developing SRMC and making the algorithmic ideas,
assumptions, and experimental targets clear enough to audit. The paper's
constant-memory score-history perspective made a useful clean-room
implementation possible, and the authors' explicit caveat about the relaxed
discrete score helped us report the Static-MNIST mismatch honestly.

## Attribution

This audit and its documentation are maintained by
[MachineLearning-Nerd](https://github.com/MachineLearning-Nerd).
