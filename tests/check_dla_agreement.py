"""Compares DLA (--correlate --csv) of one outcome on a smoothed feature table with DLA on the table it was smoothed from.
Usage: python3 check_dla_agreement.py RAW_CSV SMOOTHED_CSV OUTCOME
Passes (prints 'dla agreement ok') when the smoothed table has significant features (corrected p < .05) and its
correlations agree with the raw table's over their shared features (Pearson r of the two sets of r >= MIN_AGREEMENT).
Also prints, for each table, how many features are significant and the mean |r| of the top 20, to compare methods."""
import csv, sys
import numpy as np

MIN_AGREEMENT = 0.7
ALPHA = 0.05
TOP = 20

def readDLA(path, outcome):
    """feature -> (r, p) for outcome, from a dlatk --csv correlation file (settings, blank line, then the table)"""
    with open(path, newline='') as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('feature,'))
    rows = csv.reader(lines[start:])
    header = next(rows)
    col = header.index(outcome)
    out = dict()
    for row in rows:
        if not row: #end of the table (a sorted copy follows)
            break
        if len(row) > col + 1 and row[col] not in ('', 'nan'):
            out[row[0]] = (float(row[col]), float(row[col + 1]))
    return out

def summary(name, dla):
    rs = np.array([r for r, p in dla.values()])
    sig = sum(1 for r, p in dla.values() if p < ALPHA)
    top = np.sort(np.abs(rs))[::-1][:TOP].mean() if len(rs) else float('nan')
    print('%s: %d features, %d significant (p < %g), mean |r| of top %d = %.3f' % (name, len(rs), sig, ALPHA, TOP, top))
    return sig

rawPath, smoothPath, outcome = sys.argv[1:4]
raw, smooth = readDLA(rawPath, outcome), readDLA(smoothPath, outcome)
summary('raw     ', raw)
sig = summary('smoothed', smooth)
shared = sorted(set(raw) & set(smooth))
agreement = np.corrcoef([raw[f][0] for f in shared], [smooth[f][0] for f in shared])[0, 1] if len(shared) > 2 else float('nan')
print('agreement of r over %d shared features: %.3f (at least %g)' % (len(shared), agreement, MIN_AGREEMENT))
ok = sig > 0 and agreement >= MIN_AGREEMENT
print('dla agreement ok' if ok else 'dla agreement FAILED')
sys.exit(0 if ok else 1)
