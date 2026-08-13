# Status — Score-Repellent Monte Carlo

- Paper: arXiv 2604.22948v2, revised 2026-05-22
- Scope: bounded clean-room NumPy and small-EBM audit
- Canonical verdict: **INCONCLUSIVE**
- Local checks passing: **5/6**
- Honest negative: **C4 Static-MNIST**
- Full paper claims independently verified: **0/6**

Claim statuses:

- C0 FINITE_ACCOUNTING
- C1 PARTIAL_REPRODUCTION
- C2 PROXY_PASS
- C3 MIXED_PARTIAL
- C4 NOT_REPRODUCED
- C5 FINITE_EXACT_CHECK

The earlier “5/6 claims verified = 10 pts” label is retained only in old
history and raw run context. The canonical interpretation distinguishes
finite checks from theorem proofs and paper-scale experiments.

Core command:

~~~bash
python3 repro/src/verify.py --claims 0,1,2,3,5
~~~

See README.md and outputs/verdict.json for the evidence ledger.
