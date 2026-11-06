"""
Smoothed estimates of n-gram rates, used by FeatureRefiner.createSmoothedFeatTable (--smooth_feats).

Each estimator works on one feature at a time, vectorized over groups:
  c  -- the feature's count in each group (feature table 'value'; 0 where the group never used it)
  N  -- each group's total n-grams of the feature's n (meta table '_total{n}grams')
  p  -- the feature's background rate, sum(c) / sum(N) over the groups
With log=True the value is the posterior expected log rate E[log theta | c, N] (rates of a feature across
users and counties are close to log-normal, so this puts them on a roughly Normal scale).
"""
import numpy as np
from scipy.special import digamma, lambertw

METHODS = ('dirichlet', 'nb', 'pln', 'zinb', 'logodds_idp', 'logodds', 'hurdle')
LOG_METHODS = ('dirichlet', 'nb', 'pln', 'zinb') #methods that have a --smooth_log version

MU_GRID = np.logspace(0, 6, 121) #candidate per-group mu (pseudo-words of background), 1 to 1,000,000
A_MIN, A_MAX = 1e-3, 1e6 #clip for a feature's gamma shape (dispersion); A_MAX is effectively Poisson (no overdispersion)


## per-group mu (dirichlet, logodds_idp)

def looLogLikGrid(c, p, N, grid=MU_GRID):
    """leave-one-out log likelihood of one group's counts over a grid of mu (Zhai & Lafferty's estimator for mu):
    sum_w c_w * log((c_w - 1 + mu*p_w) / (N - 1 + mu)), for the group's observed features c (with background rates p)"""
    c = np.asarray(c, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(c) == 0:
        return np.zeros(len(grid))
    return (c[:, None] * np.log(c[:, None] - 1 + grid[None, :] * p[:, None])).sum(axis=0) - c.sum() * np.log(N - 1 + grid)

def shrinkGroupMus(groupMus, globalMu, N, minWords, muRange):
    """pulls the mu of groups with under minWords words toward the corpus-wide mu (on the log scale) and clips all
    to [globalMu/muRange, globalMu*muRange]; a group's own estimate is noisy when it has few words"""
    w = np.minimum(1.0, np.asarray(N, dtype=float) / minWords)
    logMu = w * np.log(groupMus) + (1 - w) * np.log(globalMu)
    return np.clip(np.exp(logMu), globalMu / muRange, globalMu * muRange)


## dispersion (nb, pln, zinb, hurdle)

def nbDispersion(c, N, p, weights=None):
    """method-of-moments gamma shape a for counts c ~ Poisson(N*theta), theta ~ Gamma(mean p, variance p^2/a):
    E[(c - N*p)^2 - N*p] = (N*p)^2 / a, summed over groups (optionally weighted)"""
    w = np.ones(len(c)) if weights is None else weights
    excess = (w * ((c - N * p) ** 2 - N * p)).sum()
    if excess <= 0:
        return A_MAX #no variance beyond Poisson: every group has the same rate
    return float(np.clip((w * (N * p) ** 2).sum() / excess, A_MIN, A_MAX))


## estimators: each returns one value per group

def dirichlet(c, N, p, mu, log=False):
    """Dirichlet prior (mean p, strength mu words): (c + mu*p) / (N + mu); log: E[log theta] of the Beta posterior"""
    if log:
        return digamma(c + mu * p) - digamma(N + mu)
    return (c + mu * p) / (N + mu)

def nb(c, N, p, a, log=False):
    """gamma-Poisson (negative binomial) empirical Bayes: posterior Gamma(c + a, N + a/p)"""
    b = a / p
    if log:
        return digamma(c + a) - np.log(N + b)
    return (c + a) / (N + b)

def plnParams(p, a):
    """log-normal prior with the same mean (p) and variance (p^2/a) as the gamma: log theta ~ Normal(m, s2)"""
    s2 = np.log1p(1.0 / a)
    return np.log(p) - s2 / 2, s2

def lambertwFromLog(logz):
    """principal Lambert W(z) for z = exp(logz) > 0, without overflowing when logz is large"""
    logz = np.asarray(logz, dtype=float)
    small = logz < 500
    w = np.empty_like(logz)
    w[small] = lambertw(np.exp(logz[small])).real
    if (~small).any():
        L = logz[~small]
        wl = L - np.log(L)
        for _ in range(5): #Newton on w + log(w) = L
            wl = wl - (wl + np.log(wl) - L) / (1 + 1 / wl)
        w[~small] = wl
    return w

def pln(c, N, p, a, log=False):
    """Poisson-log-normal prior, Laplace approximation in closed form:
    the posterior mode of x = log theta solves c - N*e^x - (x - m)/s2 = 0, i.e. x* = m + s2*c - W(N*s2*e^(m + s2*c));
    its variance is v = s2/(1 + W). log: x*; otherwise exp(x* + v/2)"""
    m, s2 = plnParams(p, a)
    W = lambertwFromLog(np.log(N) + np.log(s2) + m + s2 * c)
    x = m + s2 * c - W
    if log:
        return x
    return np.exp(x + s2 / (1 + W) / 2)

def plnQuadrature(c, N, p, a, log=False, points=40):
    """the same posterior by Gauss-Hermite quadrature (slow; used to check pln)"""
    m, s2 = plnParams(p, a)
    nodes, weights = np.polynomial.hermite_e.hermegauss(points)
    x = m + np.sqrt(s2) * nodes[None, :] #prior draws of log theta
    c = np.asarray(c, dtype=float)[:, None]
    N = np.asarray(N, dtype=float)[:, None]
    loglik = c * x - N * np.exp(x)
    loglik -= loglik.max(axis=1, keepdims=True)
    post = weights[None, :] * np.exp(loglik)
    post /= post.sum(axis=1, keepdims=True)
    return (post * (x if log else np.exp(x))).sum(axis=1)

def zinbFit(c, N, maxIter=100, tol=1e-6):
    """zero-inflated negative binomial: with probability pi a group never uses the feature (rate 0), otherwise
    gamma-Poisson with mean rate p and shape a. EM with a moment-based M-step; returns (pi, p, a, tau), where
    tau is each group's posterior probability of being a structural zero"""
    p = c.sum() / N.sum()
    a = nbDispersion(c, N, p)
    zero = c == 0
    nbZero = np.exp(a * np.log((a / p) / (a / p + N)))
    pi = float(np.clip(zero.mean() - nbZero.mean(), 0, 0.99))
    tau = np.zeros(len(c))
    for _ in range(maxIter):
        nbZero = np.exp(a * np.log((a / p) / (a / p + N)))
        tau = np.where(zero, pi / (pi + (1 - pi) * nbZero), 0.0) if pi > 0 else np.zeros(len(c))
        newPi = float(tau.mean())
        w = 1 - tau
        p = (w * c).sum() / (w * N).sum()
        a = nbDispersion(c, N, p, weights=w)
        if abs(newPi - pi) < tol:
            pi = newPi
            break
        pi = newPi
    return pi, p, a, tau

def zinb(c, N, pi, p, a, tau, log=False, floor=None):
    """posterior mean rate under the zero-inflated NB: the NB estimate, times (1 - tau) for groups with c = 0;
    log: log of that rate, floored (a structural zero has rate 0)"""
    rate = (c + a) / (N + a / p) * (1 - tau)
    if log:
        return np.log(np.maximum(rate, floor))
    return rate

def logoddsIDP(c, N, Cw, C, p, mu):
    """Monroe, Colaresi & Quinn's (2008) z-scored log-odds of the feature in the group versus the rest of the corpus,
    with an informative Dirichlet prior alpha_w = mu*p (alpha_0 = mu)"""
    alpha = mu * p
    delta = np.log((c + alpha) / (N + mu - c - alpha)) - np.log((Cw - c + alpha) / (C - N + mu - Cw + c - alpha))
    return delta / np.sqrt(1 / (c + alpha) + 1 / (Cw - c + alpha))

def logodds(c, N, Cw, C):
    """z-scored log-odds of the feature in the group versus the rest of the corpus, adding 0.5 to every count
    (uninformative; the baseline for logodds_idp)"""
    cells = (c + .5, N - c + .5, Cw - c + .5, C - N - Cw + c + .5)
    delta = np.log(cells[0] / cells[1]) - np.log(cells[2] / cells[3])
    return delta / np.sqrt(sum(1 / x for x in cells))

def hurdle(c, N, p, a):
    """two parts: whether the group used the feature (0/1), and the NB posterior log rate where it did
    (the feature's mean log rate elsewhere, so the second part only carries information among users)"""
    used = c > 0
    logRate = nb(c, N, p, a, log=True)
    logRate = np.where(used, logRate, logRate[used].mean() if used.any() else 0.0)
    return used.astype(float), logRate
