"""Chiral estimate of the Kato note (Ising, channel epsilon -> sigma): ||q^{L0} Y q^{L0}|| (1-q^2)^{1/16}
with chiral modules up to level nmax, in the main-code basis (code/tcsa_ising.py, Gram threshold) and in an
exactly orthonormalised basis (check). Usage: python3 ch3_chiral_estimate_check.py nmax q [q ...]"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _CODE = _os.path.join(_HERE, '..', 'code')
_sys.path.insert(0, _CODE); _sys.path.insert(0, _HERE)
import sys, numpy as np
from fractions import Fraction as F
import tcsa_ising as T
import ch3_tcsa_exact as E
nmax = int(sys.argv[1]); qs = [float(x) for x in sys.argv[2:]]
def blockmat(chiral, A, B, hA, hB, dimA, dimB, q):
    rows = []
    for n in range(nmax + 1):
        row = []
        for m in range(nmax + 1):
            if dimA(n) and dimB(m):
                row.append(q ** (float(hA) + n) * chiral(n, m) * q ** (float(hB) + m))
            else:
                row.append(np.zeros((dimA(n), dimB(m))))
        rows.append(row)
    return np.block(rows)
# main-code basis: bra sigma module, ket epsilon module
Ms, Me = T.Module("s", nmax), T.Module("e", nmax)
A_auth = lambda n, m: T.chiral_matrix(Ms, n, Me, m, T.D_SIGMA)
# exact basis
Xs, Xe = E.ExactModule("s", F(1, 16), nmax), E.ExactModule("e", F(1, 2), nmax)
A_ex = lambda n, m: E.chiral_exact(Xs, n, Xe, m, F(1, 16))
cache_a, cache_e = {}, {}
for q in qs:
    Ma = blockmat(lambda n, m: cache_a.setdefault((n, m), A_auth(n, m)), Ms, Me, F(1, 16), F(1, 2), Ms.dim, Me.dim, q)
    Mx = blockmat(lambda n, m: cache_e.setdefault((n, m), A_ex(n, m)), Xs, Xe, F(1, 16), F(1, 2), Xs.dim, Xe.dim, q)
    f = (1 - q * q) ** (1 / 16)
    print(f"nmax={nmax} q={q}: main-code basis {np.linalg.norm(Ma, 2)*f:.4f} ({Ma.shape}), exact basis {np.linalg.norm(Mx, 2)*f:.4f} ({Mx.shape})", flush=True)
