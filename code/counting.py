"""Zamolodchikov's counting argument (Chapter 4, sec:counting).

For a perturbation of a minimal model M(p,p') by the spinless primary phi_{r,s},
compare, at each spin s >= 1,
  source(s) = dim of the vacuum module at level s+1, modulo L_{-1} of level s,
  target(s) = dim of the phi module at level s, modulo L_{-1} of level s-1.
If source(s) > target(s), a conserved charge of spin s survives at first order.
Module dimensions come from the Rocha-Caridi characters of the irreducible modules.
"""
import sys

def kac(r, s, p, pp):
    from fractions import Fraction as F
    return F((r * pp - s * p) ** 2 - (pp - p) ** 2, 4 * p * pp)

def char_irr(r, s, p, pp, N):
    """Coefficients d(0..N) of q^{-Delta} chi_{r,s}(q) for M(p,p'), p<p' coprime, 1<=r<p, 1<=s<p'."""
    # Rocha-Caridi: chi = (1/prod(1-q^n)) sum_k [q^{a_k} - q^{b_k}], exponents relative to Delta_{r,s}
    from fractions import Fraction as F
    D = kac(r, s, p, pp)
    num = [0] * (N + 1)
    for k in range(-60, 61):
        a = F((2 * p * pp * k + r * pp - s * p) ** 2 - (pp - p) ** 2, 4 * p * pp) - D
        b = F((2 * p * pp * k + r * pp + s * p) ** 2 - (pp - p) ** 2, 4 * p * pp) - D
        for e, sg in ((a, 1), (b, -1)):
            assert e.denominator == 1
            if 0 <= e <= N:
                num[int(e)] += sg
    # divide by prod(1-q^n): multiply by partition generating function
    part = [1] + [0] * N
    for n in range(1, N + 1):
        for m in range(n, N + 1):
            part[m] += part[m - n]
    return [sum(num[j] * part[m - j] for j in range(m + 1)) for m in range(N + 1)]

def char_generic(null_level, N):
    """Generic c: Verma module quotient by a single null vector at level null_level (1 for the vacuum)."""
    part = [1] + [0] * N
    for n in range(1, N + 1):
        for m in range(n, N + 1):
            part[m] += part[m - n]
    return [part[m] - (part[m - null_level] if m >= null_level else 0) for m in range(N + 1)]

def counting(vac, phi, smax):
    rows = []
    for s in range(1, smax + 1):
        src = vac[s + 1] - vac[s] + (1 if s == 0 else 0)
        tgt = phi[s] - phi[s - 1]
        rows.append((s, src, tgt, src - tgt))
    return rows

if __name__ == "__main__":
    S = 22
    cases = {
        "Ising + sigma (E8)": (char_irr(1, 1, 3, 4, S + 2), char_irr(1, 2, 3, 4, S + 2)),
        "Ising + epsilon (free fermion)": (char_irr(1, 1, 3, 4, S + 2), char_irr(1, 3, 3, 4, S + 2)),
        "tricritical Ising + phi_{1,2} (E7)": (char_irr(1, 1, 4, 5, S + 2), char_irr(1, 2, 4, 5, S + 2)),
        "generic c + phi_{1,3}": (char_generic(1, S + 2), char_generic(3, S + 2)),
        "generic c + phi_{1,2}": (char_generic(1, S + 2), char_generic(2, S + 2)),
        "generic c + phi_{2,2}": (char_generic(1, S + 2), char_generic(4, S + 2)),
    }
    for name, (v, f) in cases.items():
        rows = counting(v, f, S)
        print(name, "; spins with source > target:", [s for s, a, b, d in rows if d > 0])
        print("   ", [(s, a, b) for s, a, b, d in rows])
