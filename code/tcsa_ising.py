"""Truncated conformal space approach for the critical Ising model perturbed by the spin field
(Chapter 3, Section 3.4).

    H = (2 pi / R) [ L0 + Lb0 - c/12 + 2 pi g_R B ],   g_R = B_field (R / 2 pi)^{15/8},
    B_{ab} = < a | sigma(1,1) | b >  (plane, z = 1),   restricted to spin-zero states.

Virasoro modules of c = 1/2 are built in the PBW basis L_{-k1} ... L_{-km}|Delta>, k1 >= ... >= km >= 1;
null states are removed by diagonalising the Gram matrix level by level.
Chiral matrix elements of a primary of weight D between descendants follow from
    <A|[L_n, phi(1)]|B> = (D_A - D_B + n D) <A|phi(1)|B>.

Usage:  python3 tcsa_ising.py  [Ncut]
Prints: checks, mass ratios at real field, and the collision of the two lowest levels at imaginary field.
"""
import sys
from functools import lru_cache
from fractions import Fraction as Fr
import numpy as np

c = Fr(1, 2)
WEIGHTS = {"1": Fr(0), "e": Fr(1, 2), "s": Fr(1, 16)}
D_SIGMA = Fr(1, 16)
# OPE coefficients C_{a sigma b} (two-point functions normalised to one)
C = {("1", "s"): 1.0, ("s", "1"): 1.0, ("e", "s"): 0.5, ("s", "e"): 0.5}


def partitions(n, kmax=None):
    if kmax is None:
        kmax = n
    if n == 0:
        yield ()
        return
    for k in range(min(n, kmax), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


@lru_cache(maxsize=None)
def apply_L(n, state, h):
    """L_n acting on the PBW state (k1,...,km) = L_{-k1}...L_{-km}|h>. Returns tuple of (state, coef)."""
    if len(state) == 0:
        if n > 0:
            return ()
        if n == 0:
            return (((), h),)
        return (((-n,), Fr(1)),)
    k1, rest = state[0], state[1:]
    if n < 0 and -n >= k1:
        return (((-n,) + state, Fr(1)),)
    out = {}
    # L_n L_{-k1} rest = L_{-k1} L_n rest + [L_n, L_{-k1}] rest
    for s, a in apply_L(n, rest, h):
        for s2, b in apply_L(-k1, s, h):
            out[s2] = out.get(s2, 0) + a * b
    coef = n + k1                     # (n - m) with m = -k1
    if coef != 0:
        for s, a in apply_L(n - k1, rest, h):
            out[s] = out.get(s, 0) + coef * a
    if n == k1:                       # central term c/12 (n^3 - n) delta_{n+m,0}
        out[rest] = out.get(rest, 0) + c / 12 * (n ** 3 - n)
    return tuple((s, v) for s, v in out.items() if v != 0)


def gram(level_states, h):
    G = np.zeros((len(level_states), len(level_states)))
    for i, v in enumerate(level_states):
        for j, w in enumerate(level_states):
            vec = {w: Fr(1)}
            for k in v:                   # <v|w> = <h| L_{km}...L_{k1} |w>; apply L_{k1} first
                new = {}
                for s, a in vec.items():
                    for s2, b in apply_L(k, s, h):
                        new[s2] = new.get(s2, 0) + a * b
                vec = new
            G[i, j] = float(vec.get((), 0))
    return G


class Module:
    def __init__(self, label, nmax):
        self.h = WEIGHTS[label]
        self.levels = []              # list of (PBW states, U) with U: PBW -> orthonormal (columns)
        for n in range(nmax + 1):
            st = list(partitions(n))
            G = gram(st, self.h)
            w, V = np.linalg.eigh(G)
            keep = w > 1e-10 * max(1.0, np.max(np.abs(w)))
            if np.any(w < -1e-9 * max(1.0, np.max(np.abs(w)))):
                raise ValueError("negative norm: non-unitary module")
            U = V[:, keep] / np.sqrt(w[keep])
            self.levels.append((st, U))

    def dim(self, n):
        return self.levels[n][1].shape[1]


def chiral_matrix(A, nA, B, nB, D):
    """<A, PBW state at level nA | phi(1) | B, PBW state at level nB>, phi of weight D, then orthonormalised."""
    hA, hB = A.h, B.h

    @lru_cache(maxsize=None)
    def me(v, w):
        # v: PBW tuple in module A (bra), w: PBW tuple in module B (ket)
        if len(v) > 0:
            k, vp = v[0], v[1:]   # |v> = L_{-k}|vp>, so <v| = <vp| L_k
            hvp = hA + sum(vp)
            hw = hB + sum(w)
            val = (hvp - hw + k * D) * me(vp, w)
            for s, a in apply_L(k, w, hB):
                val += a * me(vp, s)
            return val
        if len(w) > 0:
            k, wp = w[0], w[1:]   # <hA| phi L_{-k} |wp> = -<hA|[L_{-k}, phi]|wp>
            hwp = hB + sum(wp)
            return -(hA - hwp - k * D) * me((), wp)
        return Fr(1)

    stA, UA = A.levels[nA]
    stB, UB = B.levels[nB]
    M = np.array([[float(me(v, w)) for w in stB] for v in stA])
    return UA.T @ M @ UB


def build(Ncut):
    """Spin-zero states with L0 + Lb0 - (Delta + Deltabar) <= Ncut, i.e. descendant level n = nbar <= Ncut/2."""
    nmax = Ncut // 2
    mods = {a: Module(a, nmax) for a in WEIGHTS}
    blocks = []   # (label, n, dim)
    for a in WEIGHTS:
        for n in range(nmax + 1):
            d = mods[a].dim(n)
            if d:
                blocks.append((a, n, d))
    offs = np.cumsum([0] + [d * d for _, _, d in blocks])
    N = offs[-1]
    H0 = np.zeros(N)
    B = np.zeros((N, N))
    for I, (a, n, d) in enumerate(blocks):
        H0[offs[I]:offs[I + 1]] = float(2 * (WEIGHTS[a] + n) - c / 12)
    for I, (a, n, d) in enumerate(blocks):
        for J, (b, m, e) in enumerate(blocks):
            if (a, b) not in C:
                continue
            M = chiral_matrix(mods[a], n, mods[b], m, D_SIGMA)
            B[offs[I]:offs[I + 1], offs[J]:offs[J + 1]] = C[(a, b)] * np.kron(M, M)
    return H0, B


FATEEV = 4.40490858   # m1 = FATEEV * |B|^{8/15}, sigma normalised to <sigma sigma> = |x|^{-1/4}


def spectrum(H0, B, gR, k=6):
    H = np.diag(H0) + 2 * np.pi * gR * B
    ev = np.linalg.eigvals(H)
    return ev[np.argsort(ev.real)][:k]


if __name__ == "__main__" and "--tex" not in sys.argv:
    Ncut = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    H0, B = build(Ncut)
    print(f"Ncut={Ncut}: {len(H0)} spin-zero states; B symmetric: {np.allclose(B, B.T)}", file=sys.stderr)
    # real field: gaps in units of the first gap, and first gap against m1 R
    print("# real field: r = m1 R, (E1-E0)/m1, (E2-E0)/(E1-E0), (E3-E0)/(E1-E0)")
    for r in [1, 2, 3, 4, 5, 6, 7, 8]:
        gR = (r / (2 * np.pi * FATEEV)) ** (15 / 8)
        e = np.sort(spectrum(H0, B, gR).real)
        g1 = (e[1] - e[0]) * 2 * np.pi / r       # in units of m1 (E in units of 2pi/R)
        print(f"{r:4.1f} {g1:9.5f} {(e[2]-e[0])/(e[1]-e[0]):9.5f} {(e[3]-e[0])/(e[1]-e[0]):9.5f}")
    # imaginary field: the two lowest levels
    print("# imaginary field: r = m1(|B|) R, lowest two levels (units 2pi/R)")
    for r in np.linspace(0.2, 3.0, 15):
        gR = 1j * (r / (2 * np.pi * FATEEV)) ** (15 / 8)
        e = spectrum(H0, B, gR, 3)
        print(f"{r:6.3f} " + " ".join(f"{x.real:+.5f}{x.imag:+.5f}i" for x in e))


def collision(H0, B, lo=2.0, hi=2.8, tol=1e-7):
    """Smallest r = m1(|B|) R at which the two lowest levels at imaginary field B = i|B| become complex."""
    def complex_pair(r):
        e = spectrum(H0, B, 1j * (r / (2 * np.pi * FATEEV)) ** (15 / 8), 2)
        return abs(e[0].imag) > 1e-9
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if complex_pair(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def figure_data(Ncut=24):
    """TikZ coordinates for the two figures of Section 3.4."""
    H0, B = build(Ncut)
    out = {}
    rs = np.arange(1.0, 25.01, 0.5)
    levels = [[] for _ in range(6)]
    for r in rs:
        gR = (r / (2 * np.pi * FATEEV)) ** (15 / 8)
        e = np.sort(spectrum(H0, B, gR, 8).real)
        for i in range(6):
            levels[i].append((r, (e[i + 1] - e[0]) * 2 * np.pi / r))
    out["real"] = levels
    rs = np.round(np.arange(0.0, 3.001, 0.05), 3)
    rc = collision(H0, B)
    rs = np.sort(np.concatenate([rs, [rc - 1e-4, rc + 1e-4]]))
    lo, hi, im = [], [], []
    for r in rs:
        gR = 1j * (r / (2 * np.pi * FATEEV)) ** (15 / 8)
        e = spectrum(H0, B, gR, 2)
        e = sorted(e, key=lambda x: (x.real, x.imag))
        lo.append((r, e[0].real)); hi.append((r, e[1].real)); im.append((r, abs(e[0].imag)))
    out["imag"] = (lo, hi, im, rc)
    return out


def tikz(pts, fmt="{:.3f}"):
    return " ".join("(" + fmt.format(x) + "," + fmt.format(y) + ")" for x, y in pts)


def write_generated(path="generated/tcsa_ising.tex", Ncut=24):
    """Write the TikZ coordinates of the two figures of Section 3.4 and the value of r_c as LaTeX,
    to be \\input by the chapter (never edit the output by hand).  Usage: python3 tcsa_ising.py --tex"""
    d = figure_data(Ncut)
    f4 = lambda pts: " ".join(f"({x:.4f},{y:.4f})" for x, y in pts)
    lines = [f"% Generated by code/tcsa_ising.py --tex (Ncut = {Ncut}). Do not edit.",
             f"\\newcommand{{\\rcEeight}}{{{d['imag'][3]:.5f}}}"]
    lines.append("\\newcommand{\\TCSArealLevels}{%")
    for lev in d["real"]:
        pts = [(x, y) for x, y in lev if x >= 2.0]
        lines.append("\\draw[lvl] plot coordinates {" + " ".join(f"({x:.1f},{y:.3f})" for x, y in pts) + "};")
    lines.append("}")
    lo, hi, im, rc = d["imag"]
    below = [i for i, (r, _) in enumerate(lo) if r < rc]
    above = [i for i, (r, _) in enumerate(lo) if r > rc]
    lines.append("\\newcommand{\\TCSAimagLevels}{%")
    lines.append("\\draw[lv0] plot coordinates {" + f4([lo[i] for i in below]) + "};")
    lines.append("\\draw[lv1] plot coordinates {" + f4([hi[i] for i in below]) + "};")
    lines.append("\\draw[lvc] plot coordinates {" + f4([lo[i] for i in above]) + "};")
    lines.append("\\draw[lvi] plot coordinates {" + f4([(im[i][0], im[i][1]) for i in above]) + "};")
    lines.append("\\draw[lvi] plot coordinates {" + f4([(im[i][0], -im[i][1]) for i in above]) + "};")
    lines.append("}")
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__" and "--tex" in sys.argv:
    write_generated()
