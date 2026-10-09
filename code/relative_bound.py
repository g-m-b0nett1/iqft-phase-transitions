"""Is V = int phi dx relatively bounded with respect to H0 on the circle?  (Chapter 3, Section 3.3)

Kato's analytic perturbation theory needs either
  operator bound: ||V psi|| <= a ||H0 psi|| + b ||psi||        (type A), or
  form bound:     |<psi|V|psi>| <= a <psi|H0|psi> + b ||psi||^2 (type B).
With D = H0 + mu (H0 = L0 + Lb0 - c/12 shifted to be >= 0), these say that
  ||V D^{-1}||  resp.  ||D^{-1/2} V D^{-1/2}||  is finite; relative bound zero means it -> 0 as mu -> infinity.
We compute both norms in the truncated space (spin-zero sector) for increasing Ncut, together with
||P_N V|0>||^2, which converges iff X < 1/2.  Perturbations: sigma (X = 1/8) and epsilon (X = 1).
"""
import numpy as np
from fractions import Fraction as Fr
import tcsa_ising as t

FIELDS = {
    "sigma": (Fr(1, 16), {("1", "s"): 1.0, ("s", "1"): 1.0, ("e", "s"): 0.5, ("s", "e"): 0.5}),
    "epsilon": (Fr(1, 2), {("1", "e"): 1.0, ("e", "1"): 1.0, ("s", "s"): 0.5}),
}


def norms(field, Ncut, mus=(1.0, 10.0, 100.0)):
    t.D_SIGMA, t.C = FIELDS[field]
    H0, V = t.build(Ncut)
    h = H0 - H0.min()                      # H0 >= 0
    vac = np.argmin(H0)
    out = {"dim": len(H0), "vac": float(V[vac] @ V[vac])}
    for mu in mus:
        d = h + mu
        out[f"op{mu:g}"] = np.linalg.norm(V / d[None, :], 2)
        out[f"form{mu:g}"] = np.linalg.norm(V / np.sqrt(d)[:, None] / np.sqrt(d)[None, :], 2)
    return out


if __name__ == "__main__":
    for field in ("sigma", "epsilon"):
        print(f"# {field}: Ncut, dim, ||P V|0>||^2, ||V(H0+mu)^-1|| for mu=1,10,100, ||(H0+mu)^-1/2 V (H0+mu)^-1/2|| for mu=1,10,100")
        for N in (8, 12, 16, 20, 24, 28, 32):
            o = norms(field, N)
            print(f"{N:3d} {o['dim']:5d} {o['vac']:8.4f}  " + " ".join(f"{o[f'op{m:g}']:7.4f}" for m in (1, 10, 100))
                  + "   " + " ".join(f"{o[f'form{m:g}']:7.4f}" for m in (1, 10, 100)))
