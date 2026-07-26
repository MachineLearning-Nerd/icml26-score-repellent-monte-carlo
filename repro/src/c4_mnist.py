"""C4 stretch: Static-MNIST discrete EBM, GWG vs SR-GWG mode exploration.

Trains a small RBM (energy-based model) on binarized MNIST {0,1}^784, then runs
Gibbs-with-Gradients (GWG, Grathwohl 2021) and its score-repellent variant SR-GWG
(Hu et al. 2026, Appendix B.5) from a worst-case init (all chains at one '7' image).

Per-bit GWG flip log-ratio uses the first-order "took a gradient" relaxation, which the
paper (Remark 3.7) identifies with the relaxed discrete score s_i(x) = grad_i(log pi)(1-2x_i).
SR-GWG tilts the target by exp(-alpha theta^T s(x)); the per-coordinate first-order tilt
gives flip delta delta_i = s_i(x)*(1 + 2*alpha*theta_i) (see paper's relaxed-proxy caveat,
Section 4.3).  theta_n is the SRMC running average of the score.

Metrics (paper Section 4.3): cumulative KL divergence toward uniform over the 10 digit
classes (down), and batch Vendi Score (up).  Samples are classified by nearest training
class-centroid in Hamming distance.
"""

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def sigmoid(u):
    return np.where(u >= 0, 1.0 / (1.0 + np.exp(-u)), np.exp(u) / (1.0 + np.exp(u)))


def train_rbm(X, H=300, epochs=18, batch=100, lr=0.05, k=1, seed=0,
              Wmag=0.01, verbose=True):
    """Persistent contrastive divergence training of an RBM (visible marginal EBM)."""
    rng = np.random.default_rng(seed)
    N, D = X.shape
    b = np.zeros(D); c = np.zeros(H)
    W = rng.standard_normal((H, D)) * Wmag
    # persistent fantasy chain
    Xneg = X[rng.integers(0, N, size=batch)].copy()
    for ep in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, batch):
            idx = order[s:s + batch]
            xb = X[idx]
            # positive
            hp = sigmoid(c[None, :] + xb @ W.T)
            # negative (PCD, k=1) from persistent chain
            hn = sigmoid(c[None, :] + Xneg @ W.T)          # hidden prob (B,H)
            hs = (rng.uniform(size=hn.shape) < hn).astype(np.float64)   # hidden sample
            vn = sigmoid(b[None, :] + hs @ W)              # visible prob (B,D)
            Xneg = (rng.uniform(size=vn.shape) < vn).astype(np.float64)  # new persistent visible
            # gradient updates (mean over batch)
            gW = (hp.T @ xb - hn.T @ Xneg) / len(idx)
            gb = (xb - Xneg).mean(0)
            gc = (hp.mean(0) - hn.mean(0))
            W += lr * gW; b += lr * gb; c += lr * gc
        if verbose and (ep + 1) % 3 == 0:
            print(f"  rbm epoch {ep+1}/{epochs}", flush=True)
    return {"W": W, "b": b, "c": c}


def rbm_free_energy_grad(model, X):
    """grad_x log pi(x) = b + W^T sigmoid(c + W x), batched."""
    W, b, c = model["W"], model["b"], model["c"]
    h = sigmoid(c[None, :] + X @ W.T)
    return b[None, :] + h @ W          # (B, D)


def relaxed_score(model, X):
    """s_i(x) = grad_i(log pi)(1-2x_i).  Batched."""
    g = rbm_free_energy_grad(model, X)
    return g * (1.0 - 2.0 * X)


def gwg_step_batch(model, X, theta, alpha, rng, gamma_n):
    """One GWG/SR-GWG step for a batch of B chains.  Returns new X, acc, updated theta."""
    s = relaxed_score(model, X)                 # (B, D) flip log-ratios
    if alpha > 0:
        delta = s * (1.0 + 2.0 * alpha * theta[None, :])   # SR tilt (relaxed proxy)
    else:
        delta = s
    w_flip = sigmoid(delta)                      # g(exp(delta)) = sigmoid(delta)
    w_stay = np.full((X.shape[0], 1), 0.5)
    w = np.concatenate([w_flip, w_stay], axis=1)
    w = w / w.sum(1, keepdims=True)
    # sample one action per chain
    cu = np.cumsum(w, 1)
    u = rng.uniform(size=(X.shape[0], 1))
    chosen = (u < cu).argmax(1)
    moved = chosen < X.shape[1]
    Xnew = X.copy()
    rows = np.where(moved)[0]
    cols = chosen[moved]
    Xnew[rows, cols] = 1 - Xnew[rows, cols]
    # update theta_n with the relaxed score at the NEW state
    s_new = relaxed_score(model, Xnew)
    theta = theta + gamma_n * (s_new.mean(0) - theta)
    return Xnew, moved.mean(), theta


def classify_by_centroid(X, centroids):
    """Hamming-nearest class centroid (B,10 distances)."""
    # distance = number of differing bits; centroids are binary class means
    cb = (centroids > 0.5).astype(np.uint8)
    d = np.empty((X.shape[0], 10))
    for k in range(10):
        d[:, k] = (X != cb[k]).sum(1)
    return d.argmin(1)


def vendi_score(class_labels):
    """Batch Vendi Score on class labels = exp(entropy) over the 10 classes."""
    _, counts = np.unique(class_labels, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))


def cumulative_kl(history_classes):
    """Cumulative KL of all-seen-samples class histogram toward uniform(10)."""
    seen = np.asarray(history_classes).ravel()
    vals, counts = np.unique(seen, return_counts=True)
    p = np.zeros(10)
    p[vals] = counts / len(seen)
    q = np.full(10, 0.1)
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def main(steps=4000, n_chains=100, alpha=1e-4, rho=0.8, seed=7, epochs=18, H=300):
    t0 = time.time()
    data = np.load(os.path.join(OUT, "mnist_binary.npz"))
    Xtr, ytr = data["X"].astype(np.float64), data["y"]
    print(f"loaded MNIST {Xtr.shape}, training RBM H={H}...", flush=True)
    model = train_rbm(Xtr, H=H, epochs=epochs, seed=seed)
    # class centroids + a '7' init image
    cents = np.array([Xtr[ytr == k].mean(0) for k in range(10)])
    seven_imgs = Xtr[ytr == 7]
    init = np.tile(seven_imgs[0], (n_chains, 1)).copy()

    results = {}
    for label, a in [("GWG", 0.0), ("SR-GWG", alpha)]:
        rng = np.random.default_rng(seed + (1 if a else 0))
        X = init.copy()
        theta = relaxed_score(model, X).mean(0)
        kl_hist = []
        vendi_hist = []
        all_classes = []
        for n in range(steps):
            gamma_n = (n + 2) ** (-rho)
            X, acc, theta = gwg_step_batch(model, X, theta, a, rng, gamma_n)
            cls = classify_by_centroid(X, cents)
            all_classes.append(cls)
            seen = np.array(all_classes).ravel()
            kl_hist.append(cumulative_kl(seen))
            vendi_hist.append(vendi_score(cls))
        results[label] = {
            "alpha": a, "final_cumulative_kl": kl_hist[-1], "final_vendi": vendi_hist[-1],
            "kl_curve_last10": kl_hist[-10:], "vendi_curve_last10": vendi_hist[-10:],
            "classes_seen_unique": len(np.unique(all_classes)),
        }
        print(f"  {label} alpha={a}: final KL={kl_hist[-1]:.3f}  Vendi={vendi_hist[-1]:.3f}  "
              f"classes={len(np.unique(all_classes))}", flush=True)

    kl_red = 1.0 - results["SR-GWG"]["final_cumulative_kl"] / max(results["GWG"]["final_cumulative_kl"], 1e-9)
    vendi_up = results["SR-GWG"]["final_vendi"] - results["GWG"]["final_vendi"]
    out = {
        "GWG": results["GWG"], "SR-GWG": results["SR-GWG"],
        "KL_reduction_fraction": float(kl_red),
        "vendi_gain": float(vendi_up),
        "elapsed_s": time.time() - t0,
        "config": {"steps": steps, "n_chains": n_chains, "alpha": alpha, "rho": rho, "H": H, "epochs": epochs},
    }
    with open(os.path.join(OUT, "c4_mnist.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nC4 done in {out['elapsed_s']:.0f}s: KL reduction {kl_red*100:.1f}%, "
          f"Vendi {results['GWG']['final_vendi']:.2f}->{results['SR-GWG']['final_vendi']:.2f}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=18)
    main(steps=ap.parse_args().steps, alpha=ap.parse_args().alpha, epochs=ap.parse_args().epochs)
