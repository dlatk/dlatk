"""Checks the split-half reliability used by --feat_reliability_filter on simulated halves; no database needed.
Prints 'reliability math ok' when every check passes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlatk.featureRefiner import splitHalfReliability

rng = np.random.default_rng(0)
n = 100000
ok = True
for noise in (0.25, 1.0, 4.0): #each half = true score + noise; the whole (both halves) has reliability 1 / (1 + noise/2)
    t = rng.normal(size=n)
    x, y = t + rng.normal(scale=np.sqrt(noise), size=n), t + rng.normal(scale=np.sqrt(noise), size=n)
    rel = float(splitHalfReliability(x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum(), n))
    good = abs(rel - 1 / (1 + noise / 2)) < 0.02
    print('%s Spearman-Brown reliability %.3f (true %.3f)' % ('PASS' if good else 'FAIL', rel, 1 / (1 + noise / 2)))
    ok = ok and good
zero = splitHalfReliability(*(np.zeros(1) for _ in range(5)), 10)
print('%s a feature that never varies has reliability 0' % ('PASS' if zero[0] == 0 else 'FAIL'))
ok = ok and zero[0] == 0
if ok:
    print('reliability math ok')
