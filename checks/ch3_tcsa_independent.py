"""Independent TCSA for the critical Ising model perturbed by sigma (independent check of Ch. 3, Sec. 3.4).

Written from scratch as an independent check; it does not import code/tcsa_ising.py.
Differences from the main code: PBW words in increasing order L_{-k1}...L_{-km}|h>, k1<=...<=km;
chiral matrix elements reduced on the ket first (the main code reduces the bra first); the chiral
vertex operators are validated against the exact Ising conformal blocks before use.

  H = (2pi/R)[L0 + Lb0 - c/12 + 2pi g_R V],  g_R = B (R/2pi)^{15/8},  V_ab = delta_{P_a P_b} <a|sigma(1)|b>.

Usage: python3 ch3_tcsa_independent.py Ncut [Ncut ...]
"""
import sys, time
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
import mpmath as mp

c = F(1, 2)

def parts_increasing(n, kmin=1):
    if n == 0:
        yield ()
        return
    for k in range(kmin, n + 1):
        for rest in parts_increasing(n - k, k):
            yield (k,) + rest

def add(d, k, v):
    if v:
        d[k] = d.get(k, 0) + v
        if d[k] == 0:
            del d[k]

@lru_cache(maxsize=None)
def act(n, word, h):
    """L_n L_{-k1}...L_{-km}|h> (k1<=...<=km) in the canonical basis: tuple of (word, coeff)."""
    if not word:
        if n > 0:
            return ()
        if n == 0:
            return (((), h),) if h != 0 else ()
        return (((-n,), F(1)),)
    k1, rest = word[0], word[1:]
    if n < 0 and -n <= k1:
        return (((-n,) + word, F(1)),)
    out = {}
    for w, a in act(n, rest, h):            # L_{-k1} (L_n rest)
        for w2, b in act(-k1, w, h):
            add(out, w2, a * b)
    if n + k1 != 0:                         # [L_n, L_{-k1}] = (n + k1) L_{n-k1} + c/12 (n^3 - n) delta_{n,k1}
        for w, a in act(n - k1, rest, h):
            add(out, w, (n + k1) * a)
    if n == k1:
        add(out, rest, c / 12 * (n ** 3 - n))
    return tuple(out.items())

def apply_word_dagger(v, vec, h):
    """(L_{-k1}...L_{-km})^dagger = L_{km}...L_{k1}; apply L_{k1} first."""
    for k in v:
        new = {}
        for w, a in vec.items():
            for w2, b in act(k, w, h):
                add(new, w2, a * b)
        vec = new
    return vec

class Module:
    def __init__(self, h, nmax):
        self.h = F(h)
        self.basis, self.U = [], []
        for n in range(nmax + 1):
            st = list(parts_increasing(n))
            G = np.array([[float(apply_word_dagger(v, {w: F(1)}, self.h).get((), 0)) for w in st] for v in st])
            ev, vecs = np.linalg.eigh(G)
            keep = ev > 1e-9 * max(1, abs(ev).max())
            assert not np.any(ev < -1e-9 * max(1, abs(ev).max())), "negative norm"
            self.basis.append(st)
            self.U.append(vecs[:, keep] / np.sqrt(ev[keep]))
    def dim(self, n):
        return self.U[n].shape[1]

def chiral(A, nA, B, nB, D):
    """<A,level nA| phi(1) |B,level nB> for a chiral primary phi of weight D, orthonormal bases."""
    hA, hB, D = A.h, B.h, F(D)
    @lru_cache(maxsize=None)
    def me(v, w):
        if w:   # |w> = L_{-k}|w'>:  <v|phi L_{-k}|w'> = <v|L_{-k} phi|w'> - (h_v - h_w' - k D) <v|phi|w'>
            k, wp = w[0], w[1:]
            hv, hwp = hA + sum(v), hB + sum(wp)
            val = -(hv - hwp - k * D) * me(v, wp)
            for vv, a in act(k, v, hA):      # <v|L_{-k} = (L_k|v>)^dagger, real coefficients
                val += a * me(vv, wp)
            return val
        if v:   # <v'|L_k phi|hB> = (h_v' - hB + k D) <v'|phi|hB>
            k, vp = v[0], v[1:]
            return (hA + sum(vp) - hB + k * D) * me(vp, ())
        return F(1)
    M = np.array([[float(me(v, w)) for w in B.basis[nB]] for v in A.basis[nA]])
    return A.U[nA].T @ M @ B.U[nB]

def check_blocks(nmax=8):
    """Sum over intermediate states of <s|s(1)|j><j|s(1)|s> at level n must equal the z^n coefficient
    of the exact chiral Ising blocks z^{1/8}(1-z)^{-1/8} sqrt((1+sqrt(1-z))/2)  (identity channel) and
    z^{1/8-1/2}(1-z)^{-1/8} sqrt(2(1-sqrt(1-z)))  (epsilon channel)."""
    mods = {"1": Module(0, nmax), "e": Module(F(1, 2), nmax), "s": Module(F(1, 16), nmax)}
    mp.mp.dps = 30
    blk1 = lambda z: (1 - z) ** (-mp.mpf(1) / 8) * mp.sqrt((1 + mp.sqrt(1 - z)) / 2)
    # z^{-1/2} sqrt(2(1 - sqrt(1-z))) = sqrt(2/(1 + sqrt(1-z))), regular at z = 0
    blke = lambda z: (1 - z) ** (-mp.mpf(1) / 8) * mp.sqrt(2 / (1 + mp.sqrt(1 - z)))
    t1 = mp.taylor(blk1, 0, nmax)
    te = mp.taylor(blke, 0, nmax)
    err = 0
    for lab, tay in (("1", t1), ("e", te)):
        for n in range(nmax + 1):
            if mods[lab].dim(n) == 0:
                continue
            a = chiral(mods["s"], 0, mods[lab], n, F(1, 16))   # <s| s(1) |j>
            b = chiral(mods[lab], n, mods["s"], 0, F(1, 16))   # <j| s(1) |s>
            s = float((a @ b)[0, 0])
            err = max(err, abs(s - float(tay[n])))
            if abs(s - float(tay[n])) > 1e-8: print("  mismatch", lab, n, s, float(tay[n]))
    return err, mods

def build(Ncut):
    nmax = Ncut // 2
    W = {"1": F(0), "e": F(1, 2), "s": F(1, 16)}
    mods = {a: Module(W[a], nmax) for a in W}
    Cs = {("1", "s"): 1.0, ("s", "1"): 1.0, ("e", "s"): 0.5, ("s", "e"): 0.5}
    blocks = [(a, n, mods[a].dim(n)) for a in W for n in range(nmax + 1) if mods[a].dim(n)]
    off = np.cumsum([0] + [d * d for _, _, d in blocks])
    H0 = np.concatenate([np.full(d * d, float(2 * (W[a] + n) - c / 12)) for a, n, d in blocks])
    V = np.zeros((off[-1], off[-1]))
    for I, (a, n, d) in enumerate(blocks):
        for J, (b, m, e) in enumerate(blocks):
            if (a, b) in Cs:
                M = chiral(mods[a], n, mods[b], m, F(1, 16))
                V[off[I]:off[I + 1], off[J]:off[J + 1]] = Cs[(a, b)] * np.kron(M, M)
    return H0, V

KAPPA = float(4 * mp.sin(mp.pi / 5) * mp.gamma(mp.mpf(1) / 5) / (mp.gamma(mp.mpf(2) / 3) * mp.gamma(mp.mpf(8) / 15))
              * (4 * mp.pi ** 2 * mp.gamma(mp.mpf(3) / 4) * mp.gamma(mp.mpf(13) / 16) ** 2
                 / (mp.gamma(mp.mpf(1) / 4) * mp.gamma(mp.mpf(3) / 16) ** 2)) ** (mp.mpf(4) / 15))

def levels(H0, V, gR):
    ev = np.linalg.eigvals(np.diag(H0) + 2 * np.pi * gR * V)
    return ev[np.lexsort((ev.imag, ev.real))]

def rc(H0, V, lo=2.2, hi=2.6):
    for _ in range(60):
        mid = (lo + hi) / 2
        e = levels(H0, V, 1j * (mid / (2 * np.pi * KAPPA)) ** (15 / 8))
        if abs(e[0].imag) > 1e-10: hi = mid
        else: lo = mid
    return (lo + hi) / 2

if __name__ == "__main__":
    err, _ = check_blocks(8)
    print(f"chiral vertex operators vs exact Ising blocks (levels 0..8): max abs error {err:.2e}")
    for Nc in map(int, sys.argv[1:]):
        t = time.time()
        H0, V = build(Nc)
        print(f"Ncut={Nc}: {len(H0)} states, V symmetric: {np.allclose(V, V.T)}")
        print(f"  r_c = {rc(H0, V):.7f}")
        for r in (10, 20):
            e = levels(H0, V, (r / (2 * np.pi * KAPPA)) ** (15 / 8)).real
            print(f"  r={r}: 2 pi e0/r^2 = {2*np.pi*e[0]/r**2:.6f}; gaps/m1 = " + " ".join(f"{(e[i]-e[0])*2*np.pi/r:.4f}" for i in (1, 2, 3)))
        X = 1 / 8
        I2 = float(mp.pi * (mp.gamma(X / 2) / mp.gamma(1 - X / 2)) ** 2 * mp.gamma(1 - X) / mp.gamma(X))
        for r in (0.05, 0.2):
            gR = (r / (2 * np.pi * KAPPA)) ** (15 / 8)
            e = levels(H0, V, gR).real
            print(f"  r={r}: (e0+1/24)/(-pi I2 gR^2) = {(e[0]+1/24)/(-np.pi*I2*gR**2):.6f}")
        print(f"  [{time.time()-t:.0f} s]", flush=True)
