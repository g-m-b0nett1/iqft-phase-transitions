"""Do the Virasoro modules of the main code (Gram-threshold orthonormalisation in code/tcsa_ising.py) keep all states
for the minimal models used in code/relative_bound_minimal.py?  Compare kept dimensions with the characters."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CODE = _os.path.join(_HERE, '..', 'code')
import sys, importlib
from fractions import Fraction as Fr
sys.path.insert(0, _CODE); sys.path.insert(0, _HERE)
import ch4_counting_independent as C
model = sys.argv[1]; nmax = int(sys.argv[2])
import tcsa_ising as t
cases = {"tim": (4, 5, [(1, 1), (1, 2), (1, 3), (1, 4)]),
         "tcp": (6, 7, [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]),
         "m56": (5, 6, [(1, 1), (2, 1), (3, 1), (4, 1)])}
p, pp, chain = cases[model]
t.c = 1 - Fr(6 * (pp - p) ** 2, p * pp)
for r, s in chain:
    lab = f"{r}{s}"
    t.WEIGHTS[lab] = Fr((r * pp - s * p) ** 2 - (pp - p) ** 2, 4 * p * pp)
    M = t.Module(lab, nmax)
    kept = [M.dim(n) for n in range(nmax + 1)]
    ch = C.rc(r, s, p, pp)[:nmax + 1]
    bad = [n for n in range(nmax + 1) if kept[n] != ch[n]]
    print(model, (r, s), "first level with missing states:", bad[0] if bad else None, " kept/char at top level:", kept[-1], ch[-1], flush=True)
