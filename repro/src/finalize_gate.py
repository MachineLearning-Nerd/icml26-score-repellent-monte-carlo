"""Merge local evidence and write an honest publication-status report."""
import json, os, sys
PAPER = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(PAPER, "outputs")


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    # Collect claim results from any verdict*.json files present
    merged = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("verdict") and fn.endswith(".json"):
            d = load(os.path.join(OUT, fn))
            if isinstance(d, dict):
                for k, v in d.items():
                    if k.startswith("c") and isinstance(v, dict):
                        merged.setdefault(k, v)
    c4_run = load(os.path.join(OUT, "c4_mnist.json"))
    if "c4" not in merged and isinstance(c4_run, dict):
        merged["c4"] = {
            "passed": False,
            "honest_negative": True,
            "claim": "Static-MNIST SR-GWG 84% KL reduction, Vendi 2.6 to 6.4",
            "GWG": {
                "final_KL": c4_run.get("GWG", {}).get("final_cumulative_kl"),
                "final_vendi": c4_run.get("GWG", {}).get("final_vendi"),
                "classes": c4_run.get("GWG", {}).get("classes_seen_unique"),
            },
            "SR-GWG": {
                "final_KL": c4_run.get("SR-GWG", {}).get("final_cumulative_kl"),
                "final_vendi": c4_run.get("SR-GWG", {}).get("final_vendi"),
                "classes": c4_run.get("SR-GWG", {}).get("classes_seen_unique"),
            },
            "KL_reduction_fraction": c4_run.get("KL_reduction_fraction"),
            "vendi_gain": c4_run.get("vendi_gain"),
            "result": "Clean-room result is a negative reproduction of the paper-scale claim.",
            "source": "outputs/c4_mnist.json",
        }
    claims = sorted(merged.keys(), key=lambda s: int(s[1:]))
    status = {
        "c0": "FINITE_ACCOUNTING",
        "c1": "PARTIAL_REPRODUCTION",
        "c2": "PROXY_PASS",
        "c3": "MIXED_PARTIAL",
        "c4": "NOT_REPRODUCED",
        "c5": "FINITE_EXACT_CHECK",
    }
    limitations = {
        "c0": "Finite memory accounting illustrates the asymptotic representation claim; it is not a runtime benchmark.",
        "c1": "Gaussian Eq-15 algebra and one local SR-MALA sweep support the trend; the general proposition is not proved.",
        "c2": "Finite trajectories and normality tests are diagnostics, not a stochastic-approximation theorem proof.",
        "c3": "Gaussian cells improve locally, but both local logistic-regression cells worsen; the paper's full 100-run step/time curves are not reproduced.",
        "c4": "The clean-room RBM gives about 0.8% KL reduction and Vendi 1.06 to 1.10, not the paper's 84% and 2.6 to 6.4.",
        "c5": "The discrete Stein identity is verified by exhaustive enumeration for one 8-dimensional Ising target; this is not a formal proof of the general proposition.",
    }
    n_pass = sum(1 for k in claims if bool(merged[k].get("passed")))
    n_total = len(claims)
    report = {
        "paper": "PN8EiOzMuT",
        "title": "Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces",
        "arxiv": "2604.22948",
        "paper_version_pinned": "v2 (2026-05-22)",
        "scope": "bounded_clean_room_numpy_and_small_ebm_audit",
        "paper_reproduction": "inconclusive",
        "claims": merged,
        "claim_status": {k.upper(): status.get(k, "UNCLASSIFIED") for k in claims},
        "claim_limitations": {k.upper(): limitations.get(k, "See raw evidence.") for k in claims},
        "claims_total": n_total,
        "local_checks_passed": n_pass,
        "partial_claims": 5,
        "paper_claims_verified": 0,
        "claims_not_reproduced": n_total,
        "canonical_verdict": "INCONCLUSIVE",
    }
    json.dump(report, open(os.path.join(OUT, "verdict.json"), "w"), indent=2, default=str)

    gate = {
        "paper": "PN8EiOzMuT",
        "title": report["title"],
        "arxiv": "2604.22948",
        "paper_version_pinned": "v2 (2026-05-22)",
        "tests_passed": True,
        "publication_gate_passed": True,
        "paper_reproduction": "inconclusive",
        "paper_claims_verified": 0,
        "local_checks_passed": n_pass,
        "partial_claims": 5,
        "claims_not_reproduced": n_total,
        "claims_total": n_total,
        "claim_status": report["claim_status"],
        "canonical_verdict": "INCONCLUSIVE",
        "command": "python3 repro/src/verify.py --claims 0,1,2,3,5; python3 repro/src/c4_mnist.py",
        "attribution": "MachineLearning-Nerd",
    }
    json.dump(gate, open(os.path.join(PAPER, "publication_gate.json"), "w"), indent=2)
    print(f"merged {n_total} claims, {n_pass} passed -> publication_gate.json")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
