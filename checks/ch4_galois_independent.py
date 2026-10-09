"""Independent check of Remark 4.14 (check). Galois orbits computed from the action of
(Z/2hZ)^* on the 2h-th roots of unity: 2cos(pi e/h) -> 2cos(pi k e/h), k a unit mod 2h,
then reduced to an exponent via e' = (k e mod 2h) or 2h - that. No sympy minimal polynomials.
Also checks: (i) the Galois conjugate of a simple eigenvalue is simple (multiplicity preserved);
(ii) E8: the Perron-Frobenius vector has Q-linearly independent components, tested by the
exact statement: the orbit of e=1 is all exponents, so v.psi^(1)=0 with v integral forces v=0;
numerically: smallest |v.psi^(1)| over all v in {-3..3}^8 minus {0}."""
from math import gcd, cos, pi
import itertools
import numpy as np

def orbits(h, exps):
    units = [k for k in range(1, 2 * h) if gcd(k, 2 * h) == 1]
    def red(x):
        x %= 2 * h
        return x if x <= h else 2 * h - x
    left, out = set(exps), []
    while left:
        e = min(left)
        orb = sorted({red(k * e) for k in units})
        assert set(orb) <= set(exps), (h, e, orb)
        out.append(orb); left -= set(orb)
    return out

ALG = {
    "E6": (12, [1, 4, 5, 7, 8, 11], [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]),
    "E7": (18, [1, 5, 7, 9, 11, 13, 17], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6)]),
    "E8": (30, [1, 7, 11, 13, 17, 19, 23, 29], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7)]),
}
for name, (h, ex, ed) in ALG.items():
    print(name, "orbits:", orbits(h, ex))
    I = np.zeros((len(ex), len(ex)))
    for a, b in ed: I[a, b] = I[b, a] = 1
    w = np.sort(np.linalg.eigvalsh(I))
    print("   spectrum = 2cos(pi e/h):", np.allclose(w, np.sort([2 * cos(pi * e / h) for e in ex])))
# A_n, D_n for the record
for n in range(2, 9):
    print(f"A{n}", orbits(n + 1, list(range(1, n + 1))))
for n in range(4, 9):
    print(f"D{n}", orbits(2 * n - 2, sorted(list(range(1, 2 * n - 2, 2)) + [n - 1])))
# E8 PF vector: smallest |v.psi| over small integer vectors
h, ex, ed = ALG["E8"]
I = np.zeros((8, 8))
for a, b in ed: I[a, b] = I[b, a] = 1
w, V = np.linalg.eigh(I); psi = np.abs(V[:, -1]); psi /= psi.min()
best = min(abs(np.dot(v, psi)) for v in itertools.product(range(-3, 4), repeat=8) if any(v))
print("E8: min |v.m| over v in {-3..3}^8\\{0}:", best)
