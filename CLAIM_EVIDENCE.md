# Claim-to-evidence ledger

This ledger separates the paper target from the finite or small-scale
computation actually present in this repository. A finite check or partial
reproduction is not a complete paper claim.

## Evidence accounting

The audit records 8 of 12 scoped evidence points:

- C0: 1 point for finite representation-size accounting.
- C1: 2 points for Gaussian analytic and local empirical covariance trends.
- C2: 2 points for finite convergence and normality diagnostics.
- C3: 1 point for the mixed continuous-target comparison.
- C4: 1 point for preserving the negative Static-MNIST result.
- C5: 1 point for exhaustive finite Stein-identity enumeration.

The count is an audit accounting device, not a paper score.

## C0 — Constant-memory accounting

- Paper target: the running score-history vector uses O(d) memory instead of
  an empirical measure over the state space.
- Production path: repro/src/verify.py:claim_c0_memory counts a 784-dimensional
  vector against the 2^784 discrete state space.
- Output fields: theta_dim_d, empirical_measure_storage_Omega,
  SRMC_memory_O(d), and log2_storage_reduction_vs_empirical_measure.
- Status: FINITE_ACCOUNTING.
- Boundary: this is representation-size accounting, not a runtime,
  allocator, or total-kernel memory benchmark.

## C1 — Proposition 3.4 and Eq. 16 covariance scaling

- Paper target: O(1/alpha) covariance behavior for the score-repellent
  estimator.
- Production path: repro/src/verify.py:claim_c1_scaling computes the
  Gaussian alpha-dependent block analytically and estimates covariance from
  independent local SR-MALA runs.
- Inputs: correlated ten-dimensional Gaussian target, finite alpha grid,
  and local seeded trajectories.
- Output fields: analytic_eq15, empirical_MALA, tail_slope_large_alpha,
  monotone_decreasing, and reduction_alpha0_to_20.
- Status: PARTIAL_REPRODUCTION.
- Boundary: the Gaussian algebra and one local sweep support the trend, but
  general targets, kernels, assumptions, and the full proposition are absent.

## C2 — Theorem 3.3 convergence and joint CLT

- Paper target: almost-sure convergence and joint central limit behavior.
- Production path: repro/src/verify.py:claim_c2_convergence_clt runs finite
  Gaussian trajectories and checks score-history thresholds, running-mean
  behavior, Shapiro-Wilk p-values, skewness, and kurtosis.
- Inputs: N = 6000, R = 200, alpha = 2.0, ten-dimensional correlated
  Gaussian target.
- Output fields: a_convergence, b_clt, N, and R.
- Status: PROXY_PASS only.
- Boundary: finite trajectory thresholds and normality tests cannot establish
  almost-sure convergence, theorem assumptions, or the joint CLT.

## C3 — Continuous-target MSE improvement

- Paper target: lower MSE on continuous targets, up to approximately 5x.
- Production path: repro/src/verify.py:claim_c3_mse compares alpha=0 with
  SR-MALA and SR-HMC on a correlated Gaussian and synthetic Bayesian logistic
  regression.
- Output fields: gaussian_MALA, gaussian_HMC, logreg_MALA, logreg_HMC,
  best_improvement_ratio, and criterion.
- Status: MIXED_PARTIAL.
- Evidence: Gaussian MALA improves about 2.18x and Gaussian HMC about 3.10x,
  while both tested logistic cells worsen relative to baseline.
- Boundary: the paper's 100-replicate step-count and CPU-time curves are not
  reproduced; the local settings are not evidence for a universal gain.

## C4 — Static-MNIST SR-GWG quality

- Paper target: 84% KL reduction and Vendi 2.6 to 6.4.
- Production path: repro/src/c4_mnist.py trains a small clean-room RBM and
  compares 100-chain GWG and SR-GWG; outputs/c4_mnist.json is merged by
  repro/src/finalize_gate.py.
- Output fields: GWG, SR-GWG, KL_reduction_fraction, vendi_gain, and result.
- Status: NOT_REPRODUCED; honest negative retained.
- Evidence: about 0.8% KL reduction and Vendi 1.06 to 1.10, not the paper's
  reported 84% and 2.6 to 6.4.
- Boundary: the RBM, run length, score scale, and initialization differ from
  the paper's deep EBM setup. The result is a mismatch report, not a
  falsification of the paper.

## C5 — Discrete Stein identity

- Paper target: Proposition 3.6, E_pi[s_i] = 0 for the discrete score.
- Production path: repro/src/verify.py:claim_c5_discrete_stein exhaustively
  enumerates all 2^8 states of a finite Ising target and accumulates the
  weighted score means.
- Output fields: E_pi_s_mean_abs_max, E_pi_s_vector, sample_score_at_x0,
  and criterion.
- Status: FINITE_EXACT_CHECK.
- Boundary: exact for one finite 8-dimensional target, not a formal proof of
  the general proposition.

## Recheck commands

~~~text
python3 repro/src/verify.py --claims 0,1,2,3,5
python3 repro/src/c4_mnist.py --steps 3000 --alpha 1e-4 --epochs 18
python3 repro/src/finalize_gate.py
python3 verify_final.py
~~~

The finalizer preserves raw values and the C4 negative result. A status
should not be upgraded without adding paper-faithful inputs, controls, and
outputs.
