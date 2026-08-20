# Environment and reproduction entry points

## Runtime

- Python: 3.x
- Numerical dependencies: NumPy and SciPy
- Data: generated in memory for C0-C3 and C5; retained binary input for C4
- Hardware: CPU is sufficient, but C1-C4 can take minutes
- Paper contract: arXiv 2604.22948v2, revised 2026-05-22

The checked-in outputs are the evidence for the current status. There is no
author environment lockfile in this audit snapshot.

## Audit commands

~~~text
python3 repro/src/verify.py --claims 0,1,2,3,5
python3 repro/src/c4_mnist.py --steps 3000 --alpha 1e-4 --epochs 18
python3 repro/src/finalize_gate.py
python3 verify_final.py
~~~

The first command writes finite core evidence. The second writes the
Static-MNIST result. The finalizer merges all retained verdict files and
rebuilds outputs/verdict.json and publication_gate.json.

The final verifier checks repository structure, branch names, commit
attribution, required metadata, and conservative claim boundaries. It does
not rerun the scientific or EBM commands.

## Interpretation

Passing a local check means only that its finite computation completed.
MIXED_PARTIAL and NOT_REPRODUCED must remain visible even when other checks
pass.
