"""Checks the --smooth_feats estimators (dlatk/smoothing.py) on simulated counts; no database needed.
Prints 'smoothing math ok' when every check passes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlatk import smoothing as sm

rng = np.random.default_rng(0)
N = rng.integers(20, 5000, size=500).astype(float)
ok = True
def check(desc, cond):
    global ok
    print('%s %s' % ('PASS' if cond else 'FAIL', desc))
    ok = ok and bool(cond)

for a, p in [(0.3, 1e-3), (5, 1e-2)]:
    c = rng.poisson(N * rng.gamma(a, p / a, size=len(N))).astype(float)
    pHat = c.sum() / N.sum()
    aHat = sm.nbDispersion(c, N, pHat)
    check('nb dispersion near its true value (a=%g: %.3g)' % (a, aHat), abs(np.log(aHat / a)) < 0.5)
    for log in (False, True):
        lap, quad = sm.pln(c, N, pHat, aHat, log), sm.plnQuadrature(c, N, pHat, aHat, log, points=80)
        check('pln closed form vs quadrature (a=%g, log=%s): r=%.5f' % (a, log, np.corrcoef(lap, quad)[0, 1]), np.corrcoef(lap, quad)[0, 1] > 0.995)
    check('dirichlet rate between MLE and background', np.all(np.minimum(c / N, pHat) - 1e-12 <= sm.dirichlet(c, N, pHat, 500.)) and np.all(sm.dirichlet(c, N, pHat, 500.) <= np.maximum(c / N, pHat) + 1e-12))

for pi in (0.0, 0.5):
    c = rng.poisson(N * rng.gamma(2, 2e-3 / 2, size=len(N)) * (rng.random(len(N)) >= pi)).astype(float)
    piHat = sm.zinbFit(c, N)[0]
    check('zinb recovers zero inflation (pi=%g: %.3f)' % (pi, piHat), abs(piHat - pi) < 0.1)

c = rng.poisson(N * rng.gamma(0.3, 1e-3 / 0.3, size=len(N))).astype(float)
pHat, used = c.sum() / N.sum(), c > 0
aHat = sm.nbDispersion(c, N, pHat)
rate, logRate = sm.nb(c, N, pHat, aHat), sm.nb(c, N, pHat, aHat, log=True)
zg, zgLog = sm.zeroGap(rate, c), sm.zeroGap(logRate, c, log=True)
check('zero gap: unused groups 0, used groups keep their smoothed rate', np.all(zg[~used] == 0) and np.allclose(zg[used], rate[used]))
check('zero gap, log: unused groups ln 2 below the smallest used log rate',
      np.allclose(zgLog[~used], logRate[used].min() - np.log(2)) and np.allclose(zgLog[used], logRate[used]))

L = np.array([-5., 0., 10., 600., 2000.])
w = sm.lambertwFromLog(L)
check('Lambert W solves w + log w = log z', np.allclose(w + np.log(w), L))

print('smoothing math ok' if ok else 'smoothing math FAILED')
sys.exit(0 if ok else 1)
