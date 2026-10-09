"""Form and operator bounds of V = int phi dx relative to H0 for a primary phi of a Virasoro minimal model
(Chapter 3, Section 3.3; companion of relative_bound.py).

Usage:  python3 relative_bound_minimal.py MODEL
MODEL = tim   : M(4,5), phi = phi_{1,2}, Delta = 1/10, X = 1/5   (the E7 perturbation)
        tcp   : M(6,7), phi = phi_{1,2}, Delta = 1/7,  X = 2/7   (weight of the E6 perturbation; Virasoro M(6,7))
        m56   : M(5,6), phi = phi_{2,1}, Delta = 2/5,  X = 4/5   (weight of the A2 / Potts energy field)
Only the modules reached from the vacuum by repeated fusion with phi are kept (an invariant subspace of V).
All allowed OPE coefficients are set to 1: the bounds tested are channel by channel, and their finiteness
does not depend on these values.  Run each model in a separate process (the Virasoro cache depends on c).
"""
import sys
from fractions import Fraction as Fr
import numpy as np
import tcsa_ising as t


def kac(p, pp, r, s):
    return Fr((r * pp - s * p) ** 2 - (pp - p) ** 2, 4 * p * pp)


def setup(model):
    if model == "tim":
        p, pp, field, chain = 4, 5, (1, 2), [(1, 1), (1, 2), (1, 3), (1, 4)]
    elif model == "tcp":
        p, pp, field, chain = 6, 7, (1, 2), [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
    elif model == "m56":
        p, pp, field, chain = 5, 6, (2, 1), [(1, 1), (2, 1), (3, 1), (4, 1)]
    else:
        raise SystemExit(__doc__)
    t.c = 1 - Fr(6 * (pp - p) ** 2, p * pp)
    labels = [f"{r}{s}" for r, s in chain]
    t.WEIGHTS = {lab: kac(p, pp, r, s) for lab, (r, s) in zip(labels, chain)}
    t.D_SIGMA = kac(p, pp, *field)
    # consecutive members of the chain are connected by the fusion with phi
    t.C = {}
    for a, b in zip(labels, labels[1:]):
        t.C[(a, b)] = 1.0
        t.C[(b, a)] = 1.0
    return float(2 * t.D_SIGMA)


if __name__ == "__main__":
    model = sys.argv[1]
    X = setup(model)
    print(f"# {model}: c = {t.c}, X = {X:.4f}, weights {dict((k, str(v)) for k, v in t.WEIGHTS.items())}")
    print("# Ncut, dim, ||P V|0>||^2, ||V(H0+mu)^-1|| mu=1,10,100, ||(H0+mu)^-1/2 V (H0+mu)^-1/2|| mu=1,10,100")
    for N in [int(a) for a in sys.argv[2:]] or (8, 12, 16, 20, 24):
        H0, V = t.build(N)
        h = H0 - H0.min()
        vac = np.argmin(H0)
        row = [N, len(H0), float(V[vac] @ V[vac])]
        for mu in (1.0, 10.0, 100.0):
            row.append(np.linalg.norm(V / (h + mu)[None, :], 2))
        for mu in (1.0, 10.0, 100.0):
            d = np.sqrt(h + mu)
            row.append(np.linalg.norm(V / d[:, None] / d[None, :], 2))
        print(f"{row[0]:3d} {row[1]:5d} {row[2]:8.4f}  " + " ".join(f"{x:7.4f}" for x in row[3:6])
              + "   " + " ".join(f"{x:7.4f}" for x in row[6:]), flush=True)
