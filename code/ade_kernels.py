"""Sign of the minimal ADE TBA kernels phi_ab(theta) = -i d/dtheta log S_ab(theta).

Input (convention-independent, checked against blocks for A2):
    phi~_ab(w)/(2 pi) = delta_ab - 2cosh(pi w/h) [ (2cosh(pi w/h) 1 - I)^(-1) ]_ab,
    phi~(w) = int e^{i w theta} phi(theta) dtheta.
Elementary block (x) = sinh(theta/2 + i pi x/2h) / sinh(theta/2 - i pi x/2h):
    phi_(x)(theta) = -sin(pi x/h) / (cosh theta - cos(pi x/h))   (< 0 for 0 < x < h)
    phi~_(x)(w)/(2 pi) = -sinh((h-x) pi w/h) / sinh(pi w).
With t = e^{pi w/h}:  -phi~_ab/(2pi) * (t^h - t^-h) = sum_x n_x (t^{h-x} - t^{x-h}).
Exact certificate: all n_x >= 0 with x in 1..h-1  =>  phi_ab(theta) < 0 for all theta.
"""
import sympy as sp
import numpy as np

def incidence(kind, l):
    I = sp.zeros(l, l)
    def link(i, j):
        I[i, j] = I[j, i] = 1
    if kind == 'A':
        for i in range(l - 1): link(i, i + 1)
        h = l + 1
    elif kind == 'D':
        for i in range(l - 2): link(i, i + 1)   # chain 0..l-2
        link(l - 3, l - 1)                      # fork: nodes l-2 and l-1 attached to l-3
        h = 2 * l - 2
    elif kind == 'E':
        # chain 0-1-2-3-...-(l-2), extra node l-1 attached to node 2
        for i in range(l - 2): link(i, i + 1)
        link(2, l - 1)
        h = {6: 12, 7: 18, 8: 30}[l]
    return I, h

t = sp.symbols('t', positive=True)

def blocks(kind, l):
    I, h = incidence(kind, l)
    c = t + 1 / t
    M = (c * sp.eye(l) - I).inv()  # rational in t
    res = {}
    for a in range(l):
        for b in range(a, l):
            R = (1 if a == b else 0) - c * M[a, b]
            P = sp.expand(sp.cancel(-R * (t**h - t**(-h))))
            P = sp.expand(P)
            poly = sp.Poly(sp.expand(P * t**h), t)  # ordinary polynomial
            coeffs = {m - h: poly.coeff_monomial(t**m) for m in range(0, 2 * h + 1)}
            n = {}
            ok = True
            for x in range(1, h):
                cp, cm = coeffs.get(h - x, 0), coeffs.get(x - h, 0)
                if cp != -cm: ok = False
                if cp != 0: n[x] = int(cp)
            if coeffs.get(0, 0) != 0 or any(v != 0 for k, v in coeffs.items() if abs(k) >= h):
                ok = False
            res[(a + 1, b + 1)] = (n, ok)
    return I, h, res

def phi_blocks(n, h, th):
    return sum(-m * np.sin(np.pi * x / h) / (np.cosh(th) - np.cos(np.pi * x / h)) for x, m in n.items())

def phi_spectral(I, h, a, b, th):
    Inp = np.array(I.tolist(), dtype=float)
    ev, V = np.linalg.eigh(Inp)
    out = np.zeros_like(th)
    for k in range(len(ev)):
        beta = np.arccos(np.clip(ev[k] / 2, -1, 1))  # = pi s/h
        s = beta * h / np.pi
        # -h sum psi_a psi_b cot(beta) sinh((h-s) th)/sinh(h th); th>0 grid
        out += -2 * h * V[a, k] * V[b, k] / np.tan(beta) * np.sinh((h - s) * th) / np.sinh(h * th)
    return out

cases = [('A', l) for l in range(1, 11)] + [('D', l) for l in range(4, 11)] + [('E', 6), ('E', 7), ('E', 8)]
th = np.linspace(1e-3, 12, 4000)
summary = []
for kind, l in cases:
    I, h, res = blocks(kind, l)
    all_ok = all(ok for _, ok in res.values())
    all_pos = all(all(m > 0 for m in n.values()) for n, _ in res.values())
    zero = [k for k, (n, _) in res.items() if not n]
    maxdiff, maxval = 0.0, -np.inf
    for (a, b), (n, _) in res.items():
        pb = phi_blocks(n, h, th) if n else np.zeros_like(th)
        ps = phi_spectral(I, h, a - 1, b - 1, th)
        maxdiff = max(maxdiff, np.max(np.abs(pb - ps)))
        maxval = max(maxval, np.max(pb))
    summary.append((f"{kind}{l}", h, all_ok, all_pos, zero, maxdiff, maxval))
    if (kind, l) in [('A', 2), ('A', 4), ('D', 4), ('E', 6), ('E', 8)]:
        print(f"--- {kind}{l} (h={h}) block content of S_ab, (x)-blocks with multiplicity")
        for (a, b), (n, _) in res.items():
            print(f"  S_{a}{b}: " + " ".join(f"({x})" + (f"^{m}" if m != 1 else "") for x, m in sorted(n.items())))

print("\ncase  h  decomposes  all n_x>0  identically-zero  max|blocks-spectral|  max phi on grid")
for s in summary:
    print(f"{s[0]:5s} {s[1]:3d}  {str(s[2]):5s}  {str(s[3]):5s}  {str(s[4]):18s}  {s[5]:.2e}  {s[6]:.3e}")
