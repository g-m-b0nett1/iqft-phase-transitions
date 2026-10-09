"""Minimal ADE S-matrices from Zamolodchikov's identity (Chapter 5).

What it does
  1. For each simply laced algebra (A_l, D_l, E6, E7, E8) builds the incidence matrix I of the
     Dynkin diagram, the masses (Perron-Frobenius eigenvector, Ch. 4 eq:chi-ade with s = 1) and
     orders the particles by increasing mass (Ch. 4 convention).
  2. From the Fourier form of the kernels (Ch. 5, eq:phi-fourier),
         phi~_ab(w)/(2 pi) = -[ I (2 cosh(pi w/h) - I)^(-1) ]_ab ,
     obtains, in exact arithmetic (sympy), the content of every amplitude in the blocks
         (x)(theta) = sinh(theta/2 + i pi x/(2h)) / sinh(theta/2 - i pi x/(2h)),
     i.e. integers n_x >= 0 with phi_ab = sum_x n_x phi_(x)  (Lemma 5.1, exact certificate),
     and rewrites it in the blocks {x} = (x-1)(x+1) of Braden-Corrigan-Dorey-Sasaki and Dorey.
  3. Checks: S_aa(0) = -1; unitarity and crossing; Dorey's S_11, S_12, S_13 of E8
     (Exact S-matrices, hep-th/9810026, p. 33); the closed form of the A_l amplitudes; the
     D_l spinor amplitudes; the closed forms of the E8 masses; and, for every simple pole of
     positive residue, the identification of a bound state and the S-matrix bootstrap equation
     S_dc(th) = S_da(th - i eta_a) S_db(th + i eta_b) for all d, at random complex th.
  4. With --tex writes generated/e8_masses.tex and generated/e8_smatrix.tex, and with --fig
     the figure figures/kernels_e8.pdf.

Run from the repository root:  python3 code/smatrix_ade.py [--tex] [--fig]
"""
import sys, os, itertools
import numpy as np
import sympy as sp

# ---------------------------------------------------------------- Dynkin data
def dynkin(kind, l):
    """Edges (0-based), Coxeter number, exponents."""
    e = []
    if kind == 'A':
        e = [(i, i + 1) for i in range(l - 1)]; h = l + 1
        exps = list(range(1, l + 1))
    elif kind == 'D':
        e = [(i, i + 1) for i in range(l - 2)] + [(l - 3, l - 1)]; h = 2 * l - 2
        exps = sorted(list(range(1, 2 * l - 2, 2)) + [l - 1])
    elif kind == 'E':
        e = [(i, i + 1) for i in range(l - 2)] + [(2, l - 1)]
        h = {6: 12, 7: 18, 8: 30}[l]
        exps = {6: [1, 4, 5, 7, 8, 11], 7: [1, 5, 7, 9, 11, 13, 17],
                8: [1, 7, 11, 13, 17, 19, 23, 29]}[l]
    I = np.zeros((l, l), dtype=int)
    for a, b in e:
        I[a, b] = I[b, a] = 1
    return I, h, exps

def mass_order(I):
    w, v = np.linalg.eigh(I.astype(float))
    m = np.abs(v[:, -1]); m = m / m.min()
    order = sorted(range(len(m)), key=lambda k: (round(m[k], 9), k))
    return order, m[order]

# ---------------------------------------------------------------- exact block content
t = sp.symbols('t', positive=True)

def block_content(I, h):
    """n[(a,b)] = {x: n_x} for the (x)-blocks, from the Fourier form; exact."""
    l = I.shape[0]
    Is = sp.Matrix(I.tolist())
    c = t + 1 / t
    M = (c * sp.eye(l) - Is).inv()
    out = {}
    for a in range(l):
        for b in range(a, l):
            R = c * M[a, b] - (1 if a == b else 0)          # = -phi~/(2 pi)
            P = sp.expand(sp.cancel(R * (t**h - t**(-h))))
            poly = sp.Poly(sp.expand(P * t**h), t)
            co = {k - h: int(poly.coeff_monomial(t**k)) for k in range(0, 2 * h + 1)}
            n = {}
            for x in range(1, h):
                cp, cm = co.get(h - x, 0), co.get(x - h, 0)
                assert cp == -cm, "not antisymmetric"
                if cp: n[x] = cp
            assert co.get(0, 0) == 0 and all(v == 0 for k, v in co.items() if abs(k) >= h)
            out[(a, b)] = out[(b, a)] = n
    return out

def curly_content(I, h):
    """{x}-content m[(a,b)] from the sl2 decomposition of [h] C(t)^{-1}_ab = sum_k mu_k [k],
    m_x = mu_{h-x} (C(t) = (t + 1/t) 1 - I, [k] = (t^k - t^-k)/(t - 1/t)); exact.
    Asserts mu_k >= 0 and mu_k = 0 for k >= h (Lemma 5.1)."""
    l = I.shape[0]
    Is = sp.Matrix(I.tolist())
    M = ((t + 1 / t) * sp.eye(l) - Is).inv()
    qh = (t**h - t**(-h)) / (t - 1 / t)
    out = {}
    for a in range(l):
        for b in range(a, l):
            P = sp.Poly(sp.expand(sp.cancel(M[a, b] * qh) * t**(2 * h)), t)
            c = {k - 2 * h: int(P.coeff_monomial(t**k)) for k in range(0, 4 * h + 1)}
            assert all(c.get(j, 0) == c.get(-j, 0) for j in range(4 * h)), "not palindromic"
            mu = {k: c.get(k - 1, 0) - c.get(k + 1, 0) for k in range(1, 4 * h)}
            assert all(v >= 0 for v in mu.values()), (a, b, mu)
            assert all(mu[k] == 0 for k in mu if k >= h), (a, b, mu)
            out[(a, b)] = out[(b, a)] = {h - k: v for k, v in mu.items() if v}
    return out

def rel(A, B):
    """largest relative difference of two arrays (values can be large off the real axis)"""
    return float(np.max(np.abs(A - B) / np.maximum(1.0, np.abs(B))))

def blk(x, h, th):
    return np.sinh(th / 2 + 1j * np.pi * x / (2 * h)) / np.sinh(th / 2 - 1j * np.pi * x / (2 * h))

def S_curly(m, h, th):
    out = np.ones_like(th, dtype=complex)
    for x, k in m.items():
        out *= (blk(x - 1, h, th) * blk(x + 1, h, th)) ** k
    return out

def phi_block(x, h, th):
    return -np.sin(np.pi * x / h) / (np.cosh(th) - np.cos(np.pi * x / h))


def qint(k):
    return sp.Integer(0) if k == 0 else sum(t**(k - 1 - 2 * j) for j in range(k))

def closed_forms(maxl=12):
    """[h] C(t)^{-1}_ab against the closed forms of Lemma lem:qcartan (A_l and D_l); exact."""
    ok = True
    for l in range(1, maxl + 1):
        I, h, _ = dynkin('A', l)
        M = ((t + 1 / t) * sp.eye(l) - sp.Matrix(I.tolist())).inv()
        for a in range(1, l + 1):
            for b in range(a, l + 1):
                ok &= sp.simplify(M[a - 1, b - 1] * qint(h) - qint(a) * qint(h - b)) == 0
    print("A_l, l <= %d: [h]M_ab = [a][h-b] (a <= b):" % maxl, ok)
    ok = True
    for l in range(4, maxl + 1):
        I, h, _ = dynkin('D', l)
        H = l - 1
        M = ((t + 1 / t) * sp.eye(l) - sp.Matrix(I.tolist())).inv()
        s, s2 = l - 2, l - 1                          # spinor nodes (0-based)
        for a in range(1, l - 1):                     # chain nodes 1..l-2, node 1 at the end
            for b in range(a, l - 1):
                ok &= sp.simplify(M[a - 1, b - 1] * qint(h) - qint(a) * (qint(h - b) + qint(b))) == 0
            for sp_ in (s, s2):
                ok &= sp.simplify(M[a - 1, sp_] * qint(h) - qint(a) * qint(H)) == 0
        diag = sum(qint(h - 1 - 4 * i) for i in range(h) if h - 1 - 4 * i >= 1)
        off = sum(qint(h - 3 - 4 * i) for i in range(h) if h - 3 - 4 * i >= 1)
        ok &= sp.simplify(M[s, s] * qint(h) - diag) == 0
        ok &= sp.simplify(M[s2, s2] * qint(h) - diag) == 0
        ok &= sp.simplify(M[s, s2] * qint(h) - off) == 0
    print("D_l, 4 <= l <= %d: chain, chain-spinor and spinor closed forms:" % maxl, ok)

def block_fourier_check():
    """phi~_(x)(w)/(2 pi) = -sinh((h-x) pi w/h)/sinh(pi w) by quadrature (Lemma lem:block-ft)."""
    from scipy.integrate import quad
    err = 0.0
    for h, x, w in [(3, 1, 0.7), (3, 2, 1.3), (30, 7, 0.4), (30, 29, 2.0), (12, 6, 0.0)]:
        f = lambda th: np.cos(w * th) * phi_block(x, h, th)      # phi is even
        val = 2 * quad(f, 0, 80, limit=400)[0] / (2 * np.pi)
        exact = -(h - x) / h if w == 0 else -np.sinh((h - x) * np.pi * w / h) / np.sinh(np.pi * w)
        err = max(err, abs(val - exact))
    print("Fourier transform of the block kernel, max error:", err)

# ---------------------------------------------------------------- the theory
class Theory:
    def __init__(self, kind, l):
        self.name = f"{kind}{l}"
        I, h, exps = dynkin(kind, l)
        order, m = mass_order(I)
        self.I = I[np.ix_(order, order)]
        self.h, self.exps, self.m, self.l = h, exps, m, l
        self.n = block_content(self.I, h)
        self.curly = curly_content(self.I, h)

    def S(self, a, b, th):
        return S_curly(self.curly[(a, b)], self.h, np.asarray(th, dtype=complex))

    def conj(self, a):
        """antiparticle: the species b with S_ab(i pi - th) symmetric, from equal masses."""
        th = np.array([0.37 + 0.11j])
        for b in range(self.l):
            if abs(self.m[b] - self.m[a]) < 1e-9 and \
               all(rel(self.S(a, d, 1j * np.pi - th), self.S(b, d, th)) < 1e-9 for d in range(self.l)):
                return b
        raise RuntimeError("no conjugate")

def residue(T, a, b, u):
    eps = 1e-7
    return (T.S(a, b, np.array([1j * u + eps])) * eps / 1j)[0]

def fusings(T):
    """(a, b, c, u, eta_a, eta_b) for every simple forward pole (positive residue)."""
    out = []
    for a in range(T.l):
        for b in range(a, T.l):
            nb = T.n[(a, b)]
            for x, k in nb.items():
                if k != 1:
                    continue
                u = np.pi * x / T.h
                R = residue(T, a, b, u)
                if abs(R.imag) > 1e-5 * abs(R) or R.real <= 0:
                    continue
                mc2 = T.m[a]**2 + T.m[b]**2 + 2 * T.m[a] * T.m[b] * np.cos(u)
                cs = [c for c in range(T.l) if abs(T.m[c]**2 - mc2) < 1e-8]
                if not cs:
                    out.append((a, b, None, u, None, None)); continue
                mc = np.sqrt(mc2)
                # rest frame of c: m_a e^{i eta_a} + m_b e^{-i eta_b} = m_c
                eta_a = np.arccos((mc**2 + T.m[a]**2 - T.m[b]**2) / (2 * mc * T.m[a]))
                eta_b = np.arccos((mc**2 + T.m[b]**2 - T.m[a]**2) / (2 * mc * T.m[b]))
                out.append((a, b, cs, u, eta_a, eta_b))
    return out

def check_bootstrap(T, fz, rng):
    worst = 0.0
    for a, b, cs, u, ea, eb in fz:
        if cs is None:
            return np.inf
        ok_any = False
        for c in cs:     # equal masses: one of them is the bound state
            th = rng.normal(size=4) + 1j * rng.uniform(-0.3, 0.3, size=4)
            err = max(rel(T.S(d, c, th), T.S(d, a, th - 1j * ea) * T.S(d, b, th + 1j * eb))
                      for d in range(T.l))
            if err < 1e-8:
                ok_any = True
            worst_c = err
        worst = max(worst, 0.0 if ok_any else worst_c)
    return worst

def checks(T, rng):
    th = rng.normal(size=6)
    r = {}
    r['S_aa(0)=-1'] = all(abs(T.S(a, a, np.array([1e-12]))[0] + 1) < 1e-6 for a in range(T.l))
    r['unitarity'] = max(np.max(np.abs(T.S(a, b, th) * T.S(a, b, -th) - 1))
                         for a in range(T.l) for b in range(T.l)) < 1e-10
    r['|S|=1 real'] = max(np.max(np.abs(np.abs(T.S(a, b, th)) - 1))
                          for a in range(T.l) for b in range(T.l)) < 1e-10
    cr = 0.0
    for a in range(T.l):
        ab = T.conj(a)
        for b in range(T.l):
            cr = max(cr, rel(T.S(a, b, 1j * np.pi - th), T.S(ab, b, th)))
    r['crossing'] = cr < 1e-9
    # kernel = sum of block kernels, all n_x >= 0
    r['n_x>=0'] = all(k >= 0 for v in T.n.values() for k in v.values())
    # the {x}-content reproduces the kernel: n_y = m_{y-1} + m_{y+1} (y = 1..h-1)
    r['{x} vs (x)'] = all({y: v for y in range(1, T.h) if (v := T.curly[k].get(y - 1, 0) + T.curly[k].get(y + 1, 0))} == T.n[k]
                          for k in T.n)
    fz = fusings(T)
    r['fusings'] = len(fz)
    r['unexplained simple poles'] = sum(1 for f in fz if f[2] is None)
    r['bootstrap max error'] = check_bootstrap(T, fz, rng)
    return r, fz

def fmt_curly(m, h, compact=False):
    """{x} content; compact: the crossing-symmetric pairs <x> = {x}{h-x}, x <= h/2 (BCDS table 3)."""
    if compact:
        keys = sorted(x for x in m if x <= h / 2)
        assert all(m.get(x, 0) == m.get(h - x, 0) for x in m), "not crossing-pair symmetric"
        # <x> = {x}{h-x} for x < h/2, and <h/2> = {h/2}^2, as in BCDS Table 3
        return keys, {x: (m[x] // 2 if 2 * x == h else m[x]) for x in keys}
    return sorted(m), m

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    cases = [('A', l) for l in range(1, 9)] + [('D', l) for l in range(4, 9)] + [('E', 6), ('E', 7), ('E', 8)]
    theories = {}
    for kind, l in cases:
        T = Theory(kind, l)
        theories[T.name] = T
        r, fz = checks(T, rng)
        print(f"{T.name:4s} h={T.h:2d}  " + "  ".join(f"{k}: {v}" for k, v in r.items()))

    # ---- Dorey's E8 amplitudes, p. 33: S11 = {1}{11}{19}{29}, S12 = {7}{13}{17}{23}, S13 = {2}{10}{12}{18}{20}{28}
    E8 = theories['E8']
    dorey = {(0, 0): {1: 1, 11: 1, 19: 1, 29: 1}, (0, 1): {7: 1, 13: 1, 17: 1, 23: 1},
             (0, 2): {2: 1, 10: 1, 12: 1, 18: 1, 20: 1, 28: 1}}
    for k, v in dorey.items():
        print("E8 S_%d%d matches Dorey:" % (k[0] + 1, k[1] + 1), E8.curly[k] == v)
    # ---- A_l closed form: S_ab = prod_{j=0}^{min(a,b)-1} {a+b-1-2j}  (labels 1..l along the chain)
    for l in range(1, 9):
        I, h, _ = dynkin('A', l)
        n = block_content(I, h)
        ok = True
        for a in range(1, l + 1):
            for b in range(1, l + 1):
                if a + b > h:          # then use S_ab = S_{h-a, h-b} (charge conjugation)
                    continue
                want = {}
                for j in range(min(a, b)):
                    x = a + b - 1 - 2 * j
                    want[x] = want.get(x, 0) + 1
                # compare as (x) content modulo (0), (h)
                got = n[(a - 1, b - 1)]
                wn = {}
                for x, k in want.items():
                    for y in (x - 1, x + 1):
                        if 1 <= y <= h - 1:
                            wn[y] = wn.get(y, 0) + k
                ok &= (wn == got)
        print(f"A{l} closed form prod {{a+b-1-2j}} (a+b <= h): {ok}")
    # ---- D_l spinor amplitudes: S_ss = {1}{5}{9}..., S_ss' = {3}{7}...,
    #      (x)-content (2)(4)...(h-2) for both, and S_ss = -S_ss' (eq:sm-AD)
    r2 = np.random.default_rng(2)
    thr = r2.normal(size=40) + 1j * r2.uniform(-3, 3, size=40)
    for l in range(4, 13):
        I, h, _ = dynkin('D', l)
        n = block_content(I, h)
        m = curly_content(I, h)
        s, s2 = l - 2, l - 1
        want = {x: 1 for x in range(2, h - 1, 2)}
        ok_x = n[(s, s)] == want and n[(s, s2)] == want
        ok_c = (m[(s, s)] == {x: 1 for x in range(1, h, 4)}
                and m[(s, s2)] == {x: 1 for x in range(3, h, 4)}
                and m[(s2, s2)] == m[(s, s)])
        ok_sign = rel(S_curly(m[(s, s)], h, thr), -S_curly(m[(s, s2)], h, thr)) < 1e-10
        print(f"D{l} spinors: {{x}}-content {ok_c}, (x)-content (2)(4)...(h-2) {ok_x}, S_ss = -S_ss' {ok_sign}")
    closed_forms(12)
    block_fourier_check()
    # ---- E8 masses: closed forms
    c = np.cos; pi = np.pi
    closed = [1, 2 * c(pi / 5), 2 * c(pi / 30), 2 * c(7 * pi / 30) * 2 * c(pi / 5),
              2 * c(2 * pi / 15) * 2 * c(pi / 5), 2 * c(pi / 30) * 2 * c(pi / 5),
              4 * c(pi / 5) * c(7 * pi / 30) * 2 * c(pi / 5), 4 * c(pi / 5) * c(2 * pi / 15) * 2 * c(pi / 5)]
    print("E8 masses:", np.round(E8.m, 10))
    print("closed forms agree:", np.allclose(E8.m, closed, atol=1e-12))
    print("E8 fusings (a b -> c, u/pi*h):")
    for a, b, cs, u, ea, eb in fusings(E8):
        print(f"  {a+1}{b+1} -> {[x+1 for x in cs]} at u = {round(u*E8.h/np.pi)} pi/30")

    if "--tex" in sys.argv:
        os.makedirs("generated", exist_ok=True)
        names = [r"m_1", r"2\cos\tfrac{\pi}{5}\,m_1", r"2\cos\tfrac{\pi}{30}\,m_1",
                 r"2\cos\tfrac{7\pi}{30}\,m_2", r"2\cos\tfrac{2\pi}{15}\,m_2",
                 r"2\cos\tfrac{\pi}{30}\,m_2", r"4\cos\tfrac{\pi}{5}\cos\tfrac{7\pi}{30}\,m_2",
                 r"4\cos\tfrac{\pi}{5}\cos\tfrac{2\pi}{15}\,m_2"]
        with open("generated/e8_masses.tex", "w") as f:
            f.write("% Generated by code/smatrix_ade.py --tex. Do not edit.\n")
            f.write("\\newcommand{\\EeightMassRows}{%\n")
            for a in range(8):
                f.write(f"  ${a+1}$ & ${names[a]}$ & ${E8.m[a]:.10f}$ \\\\\n")
            f.write("}\n")
        with open("generated/e8_smatrix.tex", "w") as f:
            f.write("% Generated by code/smatrix_ade.py --tex. Do not edit.\n")
            f.write("% Rows: ab & <x>-content & ab & <x>-content (two columns of 18 amplitudes).\n")
            cells = []
            for a in range(8):
                for b in range(a, 8):
                    keys, mm = fmt_curly(E8.curly[(a, b)], 30, compact=True)
                    sstr = r"\ ".join(f"{x}" + (f"^{{{mm[x]}}}" if mm[x] > 1 else "") for x in keys)
                    cells.append(f"${a+1}{b+1}$ & ${sstr}$")
            half = len(cells) // 2
            f.write("\\newcommand{\\EeightSmatrixRows}{%\n")
            for i in range(half):
                f.write(f"  {cells[i]} & {cells[half + i]} \\\\\n")
            f.write("}\n")
        print("wrote generated/e8_masses.tex, generated/e8_smatrix.tex")

    if "--fig" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})
        th = np.linspace(-4, 4, 1601)
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=False)
        cols = plt.cm.viridis(np.linspace(0, 0.9, 8))
        for b in range(8):
            phi = sum(k * phi_block(x, 30, th) for x, k in E8.n[(0, b)].items())
            axs[0].plot(th, phi, color=cols[b], lw=1)
        axs[0].set_xlabel(r"$\theta$"); axs[0].set_ylabel(r"$\varphi_{1b}(\theta)$")
        axs[0].axhline(0, color="gray", lw=0.5)
        axs[0].set_title(r"E8: $\varphi_{1b}$, $b=1,\ldots,8$", fontsize=9)
        ylyl = -4 * np.sqrt(3) * np.cosh(th) / (1 + 2 * np.cosh(2 * th))
        axs[1].plot(th, ylyl, color="k", lw=1)
        axs[1].axhline(0, color="gray", lw=0.5)
        axs[1].set_xlabel(r"$\theta$"); axs[1].set_ylabel(r"$\varphi(\theta)$")
        axs[1].set_title("scaling Yang–Lee", fontsize=9)
        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/kernels_e8.pdf")
        print("wrote figures/kernels_e8.pdf")
