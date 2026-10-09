"""TCSA for Ising + sigma with an exactly orthonormalised basis (independent check of Sec. 3.4).

Both the main code (code/tcsa_ising.py) and the first independent code (ch3_tcsa_independent.py) orthonormalise the PBW basis with a
floating-point eigenvalue threshold on the Gram matrix; from level 11 on this discards genuine states
(see ch3_tcsa_statecount.py). Here the rank is found by exact rational elimination, the basis is
orthonormalised by a Cholesky factorisation in 60-digit arithmetic, and the number of states is checked
against the characters at every level. Usage: python3 ch3_tcsa_exact.py Ncut [Ncut ...]"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CODE = _os.path.join(_HERE, '..', 'code')
import sys, time
from fractions import Fraction as F
import numpy as np, mpmath as mp
sys.path.insert(0, _HERE)
import ch3_tcsa_independent as R          # act(), apply_word_dagger(), parts_increasing(), KAPPA
import ch4_counting_independent as C
mp.mp.dps = 60
CH = dict(zip("1es", C.ising_chars()))

def independent_rows(G):
    """indices of a maximal set of linearly independent rows of the rational matrix G (exact)."""
    n = len(G); rows = []; basis = []   # basis: reduced rows (dict pivot -> row)
    piv = []
    M = [list(r) for r in G]
    red = []
    for i in range(n):
        v = M[i][:]
        for (p, w) in red:
            if v[p] != 0:
                f = v[p] / w[p]
                v = [a - f * b for a, b in zip(v, w)]
        nz = next((j for j, a in enumerate(v) if a != 0), None)
        if nz is not None:
            red.append((nz, v)); rows.append(i)
    return rows

class ExactModule:
    def __init__(self, lab, h, nmax):
        self.h = F(h); self.words = []; self.U = []
        for n in range(nmax + 1):
            st = list(R.parts_increasing(n))
            G = [[R.apply_word_dagger(v, {w: F(1)}, self.h).get((), F(0)) for w in st] for v in st]
            idx = independent_rows(G)
            assert len(idx) == CH[lab][n], (lab, n, len(idx), CH[lab][n])
            sub = [st[i] for i in idx]
            Gs = mp.matrix([[mp.mpf(G[i][j].numerator) / G[i][j].denominator for j in idx] for i in idx]) if idx else None
            if idx:
                L = mp.cholesky(Gs)                 # Gs = L L^T
                U = mp.inverse(L).T                 # columns: orthonormal combinations of the PBW words in sub
            else:
                U = None
            self.words.append(sub); self.U.append(U)
    def dim(self, n):
        return len(self.words[n])

def chiral_exact(A, nA, B, nB, D):
    from functools import lru_cache
    hA, hB, D = A.h, B.h, F(D)
    @lru_cache(maxsize=None)
    def me(v, w):
        if w:
            k, wp = w[0], w[1:]
            hv, hwp = hA + sum(v), hB + sum(wp)
            val = -(hv - hwp - k * D) * me(v, wp)
            for vv, a in R.act(k, v, hA):
                val += a * me(vv, wp)
            return val
        if v:
            k, vp = v[0], v[1:]
            return (hA + sum(vp) - hB + k * D) * me(vp, ())
        return F(1)
    M = mp.matrix([[mp.mpf(me(v, w).numerator) / me(v, w).denominator for w in B.words[nB]] for v in A.words[nA]])
    out = A.U[nA].T * M * B.U[nB]
    return np.array([[float(out[i, j]) for j in range(out.cols)] for i in range(out.rows)])

def build(Ncut):
    nmax = Ncut // 2
    W = {"1": F(0), "e": F(1, 2), "s": F(1, 16)}
    mods = {a: ExactModule(a, W[a], nmax) for a in W}
    Cs = {("1", "s"): 1.0, ("s", "1"): 1.0, ("e", "s"): 0.5, ("s", "e"): 0.5}
    blocks = [(a, n, mods[a].dim(n)) for a in W for n in range(nmax + 1) if mods[a].dim(n)]
    off = np.cumsum([0] + [d * d for _, _, d in blocks])
    H0 = np.concatenate([np.full(d * d, float(2 * (W[a] + n) - R.c / 12)) for a, n, d in blocks])
    V = np.zeros((off[-1], off[-1]))
    cache = {}
    for I, (a, n, d) in enumerate(blocks):
        for J, (b, m, e) in enumerate(blocks):
            if (a, b) in Cs:
                key = (a, n, b, m)
                if key not in cache:
                    cache[key] = chiral_exact(mods[a], n, mods[b], m, F(1, 16))
                M = cache[key]
                V[off[I]:off[I + 1], off[J]:off[J + 1]] = Cs[(a, b)] * np.kron(M, M)
    return H0, V

if __name__ == "__main__":
    for Nc in map(int, sys.argv[1:]):
        t = time.time()
        H0, V = build(Nc)
        print(f"Ncut={Nc}: {len(H0)} states (exact count), V symmetric: {np.allclose(V, V.T)}", flush=True)
        print(f"  r_c = {R.rc(H0, V):.7f}")
        for r in (10, 20):
            e = R.levels(H0, V, (r / (2 * np.pi * R.KAPPA)) ** (15 / 8)).real
            print(f"  r={r}: 2 pi e0/r^2 = {2*np.pi*e[0]/r**2:.6f}; gaps/m1 = " + " ".join(f"{(e[i]-e[0])*2*np.pi/r:.4f}" for i in (1, 2, 3)))
        print(f"  [{time.time()-t:.0f} s]", flush=True)
