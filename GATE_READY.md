# Publication gate

**Ready for publication as an honest scoped audit.**

This gate confirms that the repository contains reproducible local evidence,
claim-level limitations, citation, and attribution. It does not claim that
the six paper claims are all reproduced.

- Local checks passing: 5/6
- C4: honest negative, not reproduced
- Full paper claims independently verified: 0/6
- Canonical result: outputs/verdict.json
- Gate metadata: publication_gate.json
- Core command: python3 repro/src/verify.py --claims 0,1,2,3,5
- C4 command: python3 repro/src/c4_mnist.py --steps 3000 --alpha 1e-4 --epochs 18
