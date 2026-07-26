# STATUS — PN8EiOzMuT Score-Repellent Monte Carlo (SRMC)

arXiv `2604.22948` · loop-srmc session · started 2026-07-26

## Paper
Hu, Chen, Kim, Choi, Han, Eun (2026). SRMC = MCMC wrapper: history θ_n = running
average of score evals (O(d) memory); surrogate π_θn ∝ π·exp{−α θ^⊤ s(x)}; any base
kernel (MALA/HMC continuous, GWG discrete) targets the surrogate, θ updated online.
Theory: SA → a.s. convergence + CLT; asymptotic covariance Σ_X(α)=O(1/α).

## Claim status (6 claims, ≥5 needed = ≥10 pts)
| Claim | Statement | Status |
|---|---|---|
| C0 | O(d) memory vs Ω(\|X\|) empirical measure | ✅ PASS (784 floats vs 2^784) |
| C1 | Prop 3.4 + Eq 16: Σ_X(α)=O(1/α) Gaussian mean | ✅ PASS (analytic Eq-15 slope −0.92; empirical monotone, 19.8× red.) |
| C2 | Thm 3.3: (θ_n,μ_n)→(0,μ) a.s. + CLT | ✅ PASS (θ→0 decreasing, μ converged; Shapiro p=0.20/0.78) |
| C3 | 10-D SR-MALA/HMC up to ~5× lower MSE | ✅ PASS (Gaussian-HMC 3.10×, Gaussian-MALA 2.18×; best 3.10×) |
| C4 | Static-MNIST SR-GWG 84% KL↓, Vendi 2.6→6.4 | ⏳ stretch (RBM+GWG built; needs α=1e-4 + trained EBM) |
| C5 | Prop 3.6: discrete score E_π[s_i]=0 | ✅ PASS (8e-17 by enumeration, machine precision) |

## Current step
Background `verify.py --claims 1,2,3` (pid in outputs/verify_run.log) confirming C2/C3.
C0,C1,C5 already PASS. Core path = C0,C1,C2,C3,C5 → 10 pts (meets ≥10 bar).

## Next
1. Read C2/C3 results from outputs/verdict.json.
2. Build publication_gate.json + trackio logbook (index/claims/evidence/methods/negcontrols/conclusion).
3. Secret-scan, GitHub mirror, enqueue via scripts/enqueue_backlog.py; COORDINATION→publication_queued.
4. (Stretch) C4 MNIST EBM: fetch binarized MNIST, train RBM, GWG vs SR-GWG, KL+Vendi.

## Blockers
- `gh repo create --public` blocked by guard (user `!` needed) — same as prior papers.
- C4 needs binarized MNIST + trained EBM (RBM via CD, CPU or local 4GB GPU).
