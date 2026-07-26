# Score-Repellent Monte Carlo (SRMC) — ICML 2026 reproduction

Reproduction of **Hu, Chen, Kim, Choi, Han, Eun (2026), "Score-Repellent Monte Carlo:
Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces"**
(arXiv:2604.22948, OpenReview `PN8EiOzMuT`).

SRMC is a generic MCMC wrapper: trajectory history is summarized by a **d-dimensional
running average of score evaluations** θ_n (O(d) memory), converted into a surrogate
target π_θn(x) ∝ π(x)·exp{−α·θ_n^⊤ s(x)} via an exponential score tilt. Any base kernel
(MALA/HMC for continuous, GWG for discrete) is run on the current surrogate, and θ_n is
updated online. Stochastic-approximation theory gives almost-sure convergence of
(θ_n, μ_n) → (0, μ) and a joint CLT, with the score-history covariance scaling **O(1/α)**.

## What this reproduction verifies

Clean-room numpy implementation (`repro/src/srmc.py`) of every equation
(θ-recursion Eq 1, surrogate Eq 2, SR-MH Eq 4, surrogate score Eq 5, HVP Eq 6,
SA recursion Eq 7/8, Gaussian CLT Eq 16, discrete score Eq 17) and an independent
verification harness (`repro/src/verify.py`).

| Claim | Statement | Method | Outcome |
|---|---|---|---|
| C0 | O(d) memory vs Ω(\|X\|) empirical measure | storage count | see verdict.json |
| C1 | Prop 3.4 + Eq 16: Σ_X(α)=O(1/α) | analytic Eq-15 (exact) + empirical MALA | see verdict.json |
| C2 | Thm 3.3: (θ_n,μ_n)→(0,μ) a.s. + CLT | trajectory decay + Shapiro–Wilk/QR | see verdict.json |
| C3 | 10-D SR-MALA/HMC up to ~5× lower MSE | corr-Gaussian(ρ=.9) + Bayes-LogReg | see verdict.json |
| C4 | Static-MNIST SR-GWG 84% KL↓, Vendi 2.6→6.4 | discrete EBM, GWG vs SR-GWG | see verdict.json |
| C5 | Prop 3.6: discrete score E_π[s_i]=0 | exhaustive enumeration | machine precision |

Run everything from the paper directory:

```bash
source .venv/bin/activate
python repro/src/verify.py --claims 0,1,2,3,5     # core theory claims (CPU, minutes)
python repro/src/verify.py --claims 4             # MNIST discrete-EBM (stretch)
```

All evidence (raw outputs, CSVs, figures) is in `outputs/`; the deterministic verdict is
`outputs/verdict.json`; the fail-closed gate is `publication_gate.json`.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | all anchored claims, clean-room numpy | same |
| Hardware | CPU (numpy); local 4 GB GPU only for the MNIST EBM stretch | CPU |
| Time | core < 10 min; MNIST stretch longer | — |
| Cost | $0 | $0 |
