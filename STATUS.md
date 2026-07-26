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
| C4 | Static-MNIST SR-GWG 84% KL↓, Vendi 2.6→6.4 | ❌ honest negative (clean-room RBM: ~0.8% KL red, Vendi 1.06→1.10) |
| C5 | Prop 3.6: discrete score E_π[s_i]=0 | ✅ PASS (8e-17 by enumeration, machine precision) |

## Current step
**GATE COMPLETE: 5/5 claims (C0,C1,C2,C3,C5) PASS = 10 pts.** Enqueued (backlog #166);
COORDINATION → publication_queued. Secret-scan clean; git committed (17 files, .venv excluded);
publication_gate.json fail-closed `publication_gate_passed: true`. Trackio logbook built
(index/overview/claims/evidence/methods/conclusion). Drain owns HF Space publish.

**C4 MNIST stretch DONE — honest negative.** Clean-room RBM (H=300,18ep) + GWG/SR-GWG (α=1e-4):
GWG KL=2.25/Vendi=1.06/5 classes; SR-GWG KL=2.23/Vendi=1.10/5 classes → **0.8% KL reduction**
(paper: 84%). Mechanism implemented + verified by C1–C3,C5, but EBM-specific magnitude not
reproducible without authors' deep EBM. Recorded as c4 (passed=false) in verdict.json. Gate
stays 5/6 = 10 pts (C0,C1,C2,C3,C5). Not re-enqueuing.

## Next
1. Verify drain published the HF Space (DineshAI/PN8EiOzMuT); record tags + SHA in COORDINATION.
2. gh-push: BLOCKED by public-repo guard — user must run `!` (same as all prior papers).
3. Pick next paper from CPU_GPU_STARTABLE_12pt.md (fresh pool, theory-density scorer).

## Blockers
- `gh repo create --public` blocked by guard (user `!` needed) — same as prior papers.
- C4 needs binarized MNIST + trained EBM (RBM via CD, CPU or local 4GB GPU).
