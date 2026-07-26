"""Score-Repellent Monte Carlo (SRMC) — clean-room faithful implementation.

Reproduces the algorithm and theory of Hu, Chen, Kim, Choi, Han, Eun (2026),
arXiv:2604.22948 "Score-Repellent Monte Carlo: Toward Efficient Non-Markovian
Sampler with Constant Memory in General State Spaces".

Core equations (paper numbering):
  Eq (1)  theta-recursion:  theta_{n+1} = theta_n + gamma_{n+1}(s(X_{n+1}) - theta_n),
                            gamma_n = (n+1)^{-rho},  rho in (1/2, 1].
  Eq (2)  surrogate target: pi_theta(x) proportional to pi(x) * exp{-alpha * theta^T s(x)}.
  Eq (4)  SR-MH accept:     min{1, [pi(y)q(y,x)/(pi(x)q(x,y))] * e^{-alpha theta^T[s(y)-s(x)]}}.
  Eq (5)  surrogate score:  s_theta(x) = -grad U(x) + alpha * Hess U(x) * theta
                                  = s(x) + alpha * Hess U(x) * theta.
  Eq (6)  HVP finite-diff:  Hess U(x) theta ~= (grad U(x+eps*theta) - grad U(x)) / eps.
  Eq (7/8) SA recursion:    joint vartheta_n = (theta_n, mu_n) with mu_n = E_p hat of f.
  Eq (16) Gaussian CLT:     sqrt(n)( mean_i X_i - E_p[X] ) -> N(0, Sigma_X(alpha)),
                                  Sigma_X(alpha) = V Sigma_{theta theta}(alpha) V^T.
  Prop 3.4:  Sigma_{theta theta}(alpha) = O(1/alpha), monotonically decreasing in ||.||_F.
  Thm 3.3:   vartheta_n -> (0, mu) a.s.; gamma_n^{-1/2}(vartheta_n - vartheta*) -> N(0, Sigma_vartheta).
  Eq (17) discrete score:   s_i(x) = pi(x^{(i,K-x_i)})/pi(x) - 1,  E_pi[s_i]=0  (Prop 3.6).
"""

import numpy as np


# ============================================================================
#  Continuous targets
# ============================================================================

class GaussianTarget:
    """pi = N(mu, V).  Score and Hessian are analytic (Eq 16 regime)."""

    def __init__(self, mu, V):
        self.mu = np.asarray(mu, float)
        self.V = np.asarray(V, float)
        self.Vinv = np.linalg.inv(self.V)
        self.d = len(mu)

    def U(self, x):                       # potential;  pi propto exp{-U}
        r = x - self.mu
        return 0.5 * r @ self.Vinv @ r

    def grad_U(self, x):
        return self.Vinv @ (x - self.mu)

    def score(self, x):                   # s(x) = -grad U
        return -self.Vinv @ (x - self.mu)

    def hvp_U(self, x, theta, eps=None):  # exact Hess U (=Vinv) . theta
        return self.Vinv @ theta


class BayesLogRegTarget:
    """Posterior of logistic regression with a Gaussian N(0, prior_prec^-1 I) prior.

    Features X (N, d), labels y in {0,1}.  Hessian-vector product is the
    central finite-difference approximation of Eq (6).
    """

    def __init__(self, X, y, prior_prec=1.0):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        self.N, self.d = self.X.shape
        self.prior_prec = float(prior_prec)

    def U(self, w):
        z = self.X @ w
        nll = np.sum(np.logaddexp(0.0, (1.0 - 2.0 * self.y) * z))
        return nll + 0.5 * self.prior_prec * (w @ w)

    def grad_U(self, w):
        z = self.X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        return self.X.T @ (p - self.y) + self.prior_prec * w

    def score(self, w):
        return -self.grad_U(w)

    def hvp_U(self, w, theta, eps=None):
        """Analytic Hessian-vector product  Hess U(w) theta  (exact).

        Hess U = X^T diag(p(1-p)) X + prior_prec I,  p = sigmoid(X w).
        Used by the surrogate score s_theta = s + alpha * Hess U * theta (Eq 5);
        exact here (no Eq-6 finite-difference error) for accuracy.
        """
        z = self.X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        q = p * (1.0 - p)
        return self.X.T @ (q * (self.X @ theta)) + self.prior_prec * theta


# ============================================================================
#  Base kernels run on the surrogate pi_theta
# ============================================================================

def _surrogate_score(target, x, theta, alpha, eps_hvp):
    """s_theta(x) = s(x) + alpha * Hess U(x) theta   (Eq 5)."""
    return target.score(x) + alpha * target.hvp_U(x, theta, eps_hvp)


def _log_surrogate_diff(target, x, y, theta, alpha):
    """log pi_theta(y) - log pi_theta(x) = [log pi(y)-log pi(x)] - alpha theta^T[s(y)-s(x)]."""
    logpi_diff = -target.U(y) + target.U(x)
    tilt = -alpha * (theta @ (target.score(y) - target.score(x)))
    return logpi_diff + tilt


def mala_on_surrogate(target, x, theta, alpha, tau, eps_hvp, rng, L=1):
    """One Metropolis-adjusted Langevin step targeting pi_theta."""
    s_th = _surrogate_score(target, x, theta, alpha, eps_hvp)
    y = x + 0.5 * tau * s_th + np.sqrt(tau) * rng.standard_normal(target.d)

    def logq(x0, x1, sc):                 # Langevin proposal log-density
        return -0.5 * np.sum((x1 - x0 - 0.5 * tau * sc) ** 2) / tau

    s_th_y = _surrogate_score(target, y, theta, alpha, eps_hvp)
    log_ratio = (_log_surrogate_diff(target, x, y, theta, alpha)
                 + logq(y, x, s_th_y) - logq(x, y, s_th))
    if np.log(rng.uniform()) < log_ratio:
        return y, True
    return x, False


def hmc_on_surrogate(target, x, theta, alpha, tau, eps_hvp, rng, L=10):
    """L leapfrog steps targeting pi_theta, then Metropolis accept."""
    x0 = x.copy()
    p = rng.standard_normal(target.d)
    cur_K = 0.5 * p @ p
    qx = x
    grad = _surrogate_score(target, qx, theta, alpha, eps_hvp)
    step = np.sqrt(tau)
    p = p + 0.5 * step * grad
    for ell in range(L):
        qx = qx + step * p
        grad = _surrogate_score(target, qx, theta, alpha, eps_hvp)
        if ell != L - 1:
            p = p + step * grad
    p = p + 0.5 * step * grad
    p = -p
    prop_K = 0.5 * p @ p
    log_ratio = (_log_surrogate_diff(target, x0, qx, theta, alpha)
                 - prop_K + cur_K)
    if np.log(rng.uniform()) < log_ratio:
        return qx, True
    return x0, False


# ============================================================================
#  SRMC wrapper (Algorithm 1 + Eq 7 SA recursion)
# ============================================================================

def srmc_run(target, kernel, x0, alpha, rho, N, tau, eps_hvp, rng,
             L_hmc=10, f=None, theta0=None, burn=0):
    """Run SRMC.  Returns (trajectory, theta_history, mu_history, acc_count).

    gamma_n = (n+1)^{-rho}; at 0-indexed loop step n we use gamma_{n+1}=(n+2)^{-rho}
    so that for rho=1, theta_n == (1/(n+1)) sum_{i<=n} s(X_i) (verified in tests).
    """
    d = target.d
    x = np.array(x0, float)
    theta0 = target.score(x) if theta0 is None else np.array(theta0, float)
    theta = theta0.copy()
    f = (lambda z: z) if f is None else f
    mu = f(x)

    traj = np.empty((N, d))
    thetas = np.empty((N, d))
    mus = np.empty((N, d) if hasattr(f(x), "__len__") else N)
    acc = 0
    for n in range(N):
        x, a = kernel(target, x, theta, alpha, tau, eps_hvp, rng, L_hmc)
        acc += a
        gamma = (n + 2) ** (-rho)
        s_x = target.score(x)
        theta = theta + gamma * (s_x - theta)
        mu = mu + gamma * (f(x) - mu)
        traj[n] = x
        thetas[n] = theta
        mus[n] = mu
    return traj, thetas, mus, acc


# ============================================================================
#  Discrete configuration space (Prop 3.6, Eq 17)
# ============================================================================

def discrete_score(target_logpi, x, K=1):
    """Exact discrete score s_i(x) = pi(x^{(i,K-x_i)})/pi(x) - 1  (Eq 17, binary K=1).

    target_logpi(x) -> log pi(x) (unnormalized ok).  Returns array of length d.
    For binary {0,1}^d, K=1 so the neighbour flips bit i.
    """
    x = np.asarray(x)
    d = x.shape[0]
    s = np.empty(d)
    lp_x = target_logpi(x)
    for i in range(d):
        xn = x.copy()
        xn[i] = K - x[i]
        s[i] = np.exp(target_logpi(xn) - lp_x) - 1.0
    return s


def discrete_score_mean_check(target_logpi, d, K=1, n_samples=200_000, rng=None,
                              burn=5_000, exact=False):
    """Empirically / exactly check E_pi[s_i(X)] = 0 (the discrete Stein identity)."""
    rng = np.random.default_rng(0) if rng is None else rng
    if exact and d <= 16:
        # exhaustive enumeration over {0,1}^d (binary)
        states = np.array(np.unravel_index(np.arange(2 ** d), [2] * d)).T
        logp = np.array([target_logpi(s.astype(int)) for s in states])
        w = np.exp(logp - logp.max())
        w /= w.sum()
        means = np.zeros(d)
        for k, s in enumerate(states):
            means += w[k] * discrete_score(target_logpi, s.astype(int), K=K)
        return means
    # Monte-Carlo: independent draws approximating pi via Gibbs is target-specific;
    # callers pass a sampler.  This helper is used by the small exhaustive case above.
    return None


# ============================================================================
#  Discrete samplers: Gibbs-with-Gradients (GWG) and SR-GWG (Appendix B.5)
# ============================================================================

def _gwg_flip_deltas(logpi, x):
    """For each bit i: delta_i = log pi(x) - log pi(x^{i})  (energy of flipping i).

    x is binary {0,1}^d.  Returns delta array of length d and the per-bit flipped
    states' log pi via a vectorised neighbour evaluation when the caller provides a
    batched log-density; otherwise this scalar helper is used for small d.
    """
    d = x.shape[0]
    lp_x = logpi(x)
    delta = np.empty(d)
    for i in range(d):
        xn = x.copy(); xn[i] = 1 - x[i]
        delta[i] = lp_x - logpi(xn)
    return delta


def gwg_step(logpi, x, theta=None, alpha=0.0, relaxed_score=None, rng=None):
    """One GWG (or SR-GWG) step on the (tilted) target.

    Zanella locally-balanced weight g(t)=t/(1+t).  For SR-GWG the per-bit ratio is
    r_i = [pi(x^{i})/pi(x)] * exp(-alpha * theta^T[s(x^{i})-s(x)]).

    `relaxed_score(x)` -> per-coordinate relaxed score proxy s(x) in R^d (used by the
    paper's Static-MNIST SR-GWG).  If None and alpha>0, the exact discrete score
    (Eq 17) is computed from logpi.
    """
    rng = np.random.default_rng() if rng is None else rng
    d = x.shape[0]
    lp_x = logpi(x)
    # per-bit ratio pi(x^i)/pi(x) = exp(-delta_i)
    ratios = np.empty(d)
    sx = None
    if alpha > 0.0:
        sx = relaxed_score(x) if relaxed_score is not None else _exact_disc_score_from_logpi(logpi, x)
    for i in range(d):
        xn = x.copy(); xn[i] = 1 - x[i]
        lp_n = logpi(xn)
        r = np.exp(lp_n - lp_x)
        if alpha > 0.0:
            sn = relaxed_score(xn) if relaxed_score is not None else _exact_disc_score_from_logpi(logpi, xn)
            r *= np.exp(-alpha * (theta @ (sn - sx)))
        ratios[i] = r
    w = ratios / (1.0 + ratios)          # flip weights g(r)=r/(1+r)
    w_stay = 0.5                          # g(1)=1/2
    w_all = np.concatenate([w, [w_stay]])
    w_all = w_all / w_all.sum()
    j = rng.choice(d + 1, p=w_all)
    if j == d:
        return x.copy(), False
    xn = x.copy(); xn[j] = 1 - x[j]
    return xn, True


def _exact_disc_score_from_logpi(logpi, x):
    """Exact discrete score s_i(x)=pi(x^{i})/pi(x)-1  (Eq 17, binary)."""
    d = x.shape[0]; lp_x = logpi(x); s = np.empty(d)
    for i in range(d):
        xn = x.copy(); xn[i] = 1 - x[i]
        s[i] = np.exp(logpi(xn) - lp_x) - 1.0
    return s
