"""Relative-bound norms of Remark 3.4 (code/relative_bound_minimal.py) recomputed in an exactly
orthonormalised basis (exact rank by rational elimination, 60-digit Cholesky), for comparison with the
main code's Gram-threshold basis, which loses states from level 8-10 on (ch3_relbound_basischeck.py).
Same definitions as the main code: all allowed OPE coefficients 1 along the fusion chain, spin-zero states,
h = H0 - min H0, norms ||V (h+mu)^-1|| and ||(h+mu)^-1/2 V (h+mu)^-1/2||, and ||P V|0>||^2.
Usage: python3 ch3_relbound_exact.py MODEL Ncut [Ncut ...]   (MODEL = m56 | tim | tcp | ising)"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CODE = _os.path.join(_HERE, '..', 'code')
import sys
from fractions import Fraction as F
import numpy as np, mpmath as mp
sys.path.insert(0, _HERE)
import ch3_tcsa_independent as R
import ch4_counting_independent as C
import ch3_tcsa_exact as E
mp.mp.dps = 60
cases = {"tim": (4, 5, (1, 2), [(1, 1), (1, 2), (1, 3), (1, 4)]),
         "tcp": (6, 7, (1, 2), [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]),
         "m56": (5, 6, (2, 1), [(1, 1), (2, 1), (3, 1), (4, 1)]),
         "ising": (3, 4, (1, 2), [(1, 1), (1, 2), (1, 3)])}
model = sys.argv[1]; p, pp, field, chain = cases[model]
c = 1 - F(6 * (pp - p) ** 2, p * pp)
R.c = c; E.R.c = c; R.act.cache_clear()
kac = lambda r, s: F((r * pp - s * p) ** 2 - (pp - p) ** 2, 4 * p * pp)
labs = [f"{r}{s}" for r, s in chain]
W = {l: kac(r, s) for l, (r, s) in zip(labs, chain)}
E.CH = {l: C.rc(r, s, p, pp) for l, (r, s) in zip(labs, chain)}
D = kac(*field)
if model == "ising":
    pairs = {("11", "12"), ("12", "11"), ("13", "12"), ("12", "13")}
else:
    pairs = {(a, b) for a, b in zip(labs, labs[1:])} | {(b, a) for a, b in zip(labs, labs[1:])}
for Nc in map(int, sys.argv[2:]):
    nmax = Nc // 2
    mods = {l: E.ExactModule(l, W[l], nmax) for l in labs}
    blocks = [(a, n, mods[a].dim(n)) for a in labs for n in range(nmax + 1) if mods[a].dim(n)]
    off = np.cumsum([0] + [d * d for _, _, d in blocks])
    H0 = np.concatenate([np.full(d * d, float(2 * (W[a] + n) - c / 12)) for a, n, d in blocks])
    V = np.zeros((off[-1], off[-1]))
    for I, (a, n, d) in enumerate(blocks):
        for J, (b, m, e) in enumerate(blocks):
            if (a, b) in pairs:
                M = E.chiral_exact(mods[a], n, mods[b], m, D)
                V[off[I]:off[I + 1], off[J]:off[J + 1]] = np.kron(M, M)
    h = H0 - H0.min(); vac = int(np.argmin(H0))
    row = [Nc, len(H0), float(V[vac] @ V[vac])]
    for mu in (1.0, 10.0, 100.0):
        row.append(np.linalg.norm(V / (h + mu)[None, :], 2))
    for mu in (1.0, 10.0, 100.0):
        dd = np.sqrt(h + mu); row.append(np.linalg.norm(V / dd[:, None] / dd[None, :], 2))
    print(f"{model} exact: {row[0]:3d} {row[1]:5d} {row[2]:8.4f}  " + " ".join(f"{x:7.4f}" for x in row[3:6])
          + "   " + " ".join(f"{x:7.4f}" for x in row[6:]), flush=True)
