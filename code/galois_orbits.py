"""Galois orbits of the exponents and masses of the E-type ADE theories (Chapter 4, rem:galois).

For each exponent e of a simply laced algebra with Coxeter number h, the eigenvalue
lambda_e = 2 cos(pi e / h) of the incidence matrix is an algebraic number. Two exponents
are in the same orbit when their eigenvalues have the same minimal polynomial over Q.
By rem:galois, condition (C3) needs charges of infinitely many spins in one class of each
orbit. The script also prints the masses (Perron-Frobenius eigenvector, m_1 = 1), to show
which particles lie above the threshold 2 m_1, and checks with PSLQ that no small integer
relation exists among the E8 masses (a numerical sanity check of the proved statement).
"""
import numpy as np
import sympy as sp
import mpmath as mp

ALGEBRAS = {
    # name: (h, exponents, edges of the Dynkin diagram)
    "E6": (12, [1, 4, 5, 7, 8, 11], [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]),
    "E7": (18, [1, 5, 7, 9, 11, 13, 17], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6)]),
    "E8": (30, [1, 7, 11, 13, 17, 19, 23, 29], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7)]),
}

def orbits(h, exps):
    x = sp.symbols("x")
    res = {}
    for e in exps:
        p = sp.minimal_polynomial(2 * sp.cos(sp.pi * sp.Integer(e) / h), x)
        res.setdefault(p, []).append(e)
    return list(res.values())

def masses(n, edges):
    I = np.zeros((n, n))
    for a, b in edges:
        I[a, b] = I[b, a] = 1
    w, v = np.linalg.eigh(I)
    m = np.abs(v[:, -1])
    return sorted(m / m.min()), w[-1]

if __name__ == "__main__":
    mp.mp.dps = 60
    for name, (h, exps, edges) in ALGEBRAS.items():
        m, lam = masses(len(exps), edges)
        print(name, "orbits:", orbits(h, exps))
        print("   largest eigenvalue", lam, "= 2cos(pi/h)", 2 * np.cos(np.pi / h))
        print("   masses / m_1:", np.round(m, 6), " above 2 m_1:", [i + 1 for i, x in enumerate(m) if x > 2])
    M = mp.matrix(8, 8)
    for a, b in ALGEBRAS["E8"][2]:
        M[a, b] = M[b, a] = 1
    E, V = mp.eigsy(M)
    j = max(range(8), key=lambda k: E[k])
    print("E8 integer relation among masses (PSLQ, |coeff| <= 1e6):",
          mp.pslq([abs(V[k, j]) for k in range(8)], maxcoeff=10**6, maxsteps=10**6))
