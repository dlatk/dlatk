"""Checks the split-half, negative binomial and per-group reliabilities used by --feat_reliability_filter and --create_reliability_table on simulated halves; no database needed.
Prints 'reliability math ok' when every check passes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlatk.featureRefiner import splitHalfFromSums, nbReliability, groupReliability

rng = np.random.default_rng(0)
n = 100000
ok = True
for noise in (0.25, 1.0, 4.0): #each half = true score + noise; the whole (both halves) has reliability 1 / (1 + noise/2)
    t = rng.normal(size=n)
    x, y = t + rng.normal(scale=np.sqrt(noise), size=n), t + rng.normal(scale=np.sqrt(noise), size=n)
    rel = float(splitHalfFromSums(x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum(), n))
    good = abs(rel - 1 / (1 + noise / 2)) < 0.02
    print('%s Spearman-Brown reliability %.3f (true %.3f)' % ('PASS' if good else 'FAIL', rel, 1 / (1 + noise / 2)))
    ok = ok and good
zero = splitHalfFromSums(*(np.zeros(1) for _ in range(5)), 10)
print('%s a feature that never varies has reliability 0' % ('PASS' if zero[0] == 0 else 'FAIL'))
ok = ok and zero[0] == 0

# negative binomial reliability: counts c ~ Poisson(N * theta), theta ~ Gamma(shape a, mean p); true rates vary with
# variance p^2/a and a group with N words adds sampling variance p/N, so the reliability across groups is
# (p^2/a) / (p^2/a + p * mean(1/N)), which should be recovered from the counts
N = rng.integers(5, 400, size=20000).astype(float)
for a, p in ((0.5, 0.01), (5.0, 0.002)):
    c = rng.poisson(N * rng.gamma(a, p / a, size=len(N))).astype(float)
    rel = float(nbReliability(np.array([c.sum()]), np.array([(c * c).sum()]), np.array([(c * N).sum()]), N.sum(),
                              (N * N).sum(), (1 / N).mean())[0])
    true = (p * p / a) / (p * p / a + p * (1 / N).mean())
    good = abs(rel - true) < 0.03
    print('%s negative binomial reliability %.3f (true %.3f; a=%g, p=%g)' % ('PASS' if good else 'FAIL', rel, true, a, p))
    ok = ok and good

# per-group reliability (--create_reliability_table): a group with the harmonic mean total has the across-group
# reliability; under the negative binomial model it is (p^2/a) / (p^2/a + p/N); negative reliabilities give 0
h = 1 / (1 / N).mean()
good = abs(float(groupReliability(0.4, h, h)) - 0.4) < 1e-12 and float(groupReliability(-0.2, h, h)) == 0 and float(groupReliability(0.4, 0.0, h)) == 0
print('%s per-group reliability is the across-group reliability at the harmonic mean total, 0 for negative or no n-grams' % ('PASS' if good else 'FAIL'))
ok = ok and good
a, p = 0.5, 0.01
between = p * p / a
acrossRel = between / (between + p * (1 / N).mean())
err = np.abs(groupReliability(acrossRel, N, h) - between / (between + p / N)).max()
good = err < 1e-9
print('%s per-group reliability matches the negative binomial (p^2/a) / (p^2/a + p/N) (max difference %.1g)' % ('PASS' if good else 'FAIL', err))
ok = ok and good
# simulated groups of varying size, noise variance k/N: per-group reliability is exactly the true tau^2 / (tau^2 + k/N)
# given the true across-group reliability, and tracks it given the split-half estimate (within its sampling error)
tau2, k = 1.0, 200.0
err = np.abs(groupReliability(tau2 / (tau2 + k * (1 / N).mean()), N, h) - tau2 / (tau2 + k / N)).max()
good = err < 1e-9
print('%s per-group reliability from the true across-group reliability is exact over group sizes (max difference %.1g)' % ('PASS' if good else 'FAIL', err))
ok = ok and good
t = rng.normal(size=len(N))
x, y = (t + rng.normal(size=len(N)) * np.sqrt(2 * k / N) for _ in range(2)) #halves: twice the whole's noise
rel = float(splitHalfFromSums(x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum(), len(N)))
err = np.abs(groupReliability(rel, N, h) - tau2 / (tau2 + k / N)).max()
good = err < 0.06
print('%s per-group reliability from split-half tracks the true reliability over group sizes %d to %d (max difference %.3f)' % ('PASS' if good else 'FAIL', N.min(), N.max(), err))
ok = ok and good
if ok:
    print('reliability math ok')
