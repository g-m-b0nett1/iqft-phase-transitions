"""Independent recomputation of the counting tables of Sec. 4.2 (does not import code/counting.py).

Ising characters from the free-fermion q-products (not from the Rocha-Caridi formula):
  chi_0 + chi_1/2 = prod_{n>=0}(1+q^{n+1/2}),  chi_0 - chi_1/2 = prod_{n>=0}(1-q^{n+1/2}),
  chi_1/16 = prod_{n>=1}(1+q^n)   (powers of q relative to the highest weight).
Other minimal models: Feigin-Fuchs/Rocha-Caridi character written independently as an
alternating sum over the embedding diagram, cross-checked against the Ising products.
Generic c: Verma character minus the submodule of the first null vector.
"""
from fractions import Fraction as F

N = 40

def mul(a, b):
    c = [0] * (N + 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b[:N + 1 - i]):
                c[i + j] += x * y
    return c

def partitions():
    p = [1] + [0] * N
    for n in range(1, N + 1):
        for m in range(n, N + 1):
            p[m] += p[m - n]
    return p

P = partitions()

def ising_chars():
    # work in half-integer steps: index 2*level
    M = 2 * N + 2
    plus = [1] + [0] * M; minus = [1] + [0] * M
    for n in range(0, N + 1):
        k = 2 * n + 1          # exponent n+1/2 in units of 1/2
        newp = plus[:]; newm = minus[:]
        for i in range(M + 1 - k):
            newp[i + k] += plus[i]; newm[i + k] -= minus[i]
        plus, minus = newp, newm
    chi0 = [(plus[2 * L] + minus[2 * L]) // 2 for L in range(N + 1)]
    chie = [(plus[2 * L + 1] - minus[2 * L + 1]) // 2 for L in range(N + 1)]   # relative to weight 1/2
    chis = [1] + [0] * N
    for n in range(1, N + 1):
        new = chis[:]
        for i in range(N + 1 - n):
            new[i + n] += chis[i]
        chis = new
    return chi0, chie, chis

def rc(r, s, p, pp):
    """Irreducible character of M(p,p') via the alternating sum over k of Verma characters
    at weights h_{r+2kp', s}... written as q^{(2pp'k + r p' - s p)^2 ...} standard form."""
    def h(rr, ss):
        return F((rr * pp - ss * p) ** 2 - (pp - p) ** 2, 4 * p * pp)
    h0 = h(r, s)
    num = [0] * (N + 1)
    for k in range(-50, 51):
        for e, sg in ((h(r + 2 * k * p, s) - h0, 1), (h(r + 2 * k * p, -s) - h0, -1)):
            if e.denominator != 1:
                raise ValueError
            if 0 <= e <= N:
                num[int(e)] += sg
    return mul(num, P)

def generic(null):
    return [P[m] - (P[m - null] if m >= null else 0) for m in range(N + 1)]

def count(vac, phi, smax=36):
    return [(s, vac[s + 1] - vac[s], phi[s] - phi[s - 1]) for s in range(1, smax + 1)]

if __name__ == "__main__":
    chi0, chie, chis = ising_chars()
    assert chi0 == rc(1, 1, 3, 4) and chie == rc(1, 3, 3, 4) and chis == rc(1, 2, 3, 4), "Ising check failed"
    print("Ising q-products agree with the alternating-sum characters up to level", N)
    cases = {
        "E8: M(3,4)+phi12": (chi0, chis),
        "free fermion: M(3,4)+phi13": (chi0, chie),
        "E7: M(4,5)+phi12": (rc(1, 1, 4, 5), rc(1, 2, 4, 5)),
        "E6: M(6,7)+phi12 (diagonal modules)": (rc(1, 1, 6, 7), rc(1, 2, 6, 7)),
        "scaling Yang-Lee: M(2,5)+phi12": (rc(1, 1, 2, 5), rc(1, 2, 2, 5)),
        "A2/Potts: M(5,6)+phi21": (rc(1, 1, 5, 6), rc(2, 1, 5, 6)),
        "generic phi13": (generic(1), generic(3)),
        "generic phi12": (generic(1), generic(2)),
        "generic phi22": (generic(1), generic(4)),
    }
    for name, (v, f) in cases.items():
        rows = count(v, f)
        print(f"{name}: surplus>0 at", [s for s, a, b in rows if a > b])
        if name.startswith("E8"):
            print("   s=23,29:", [(s, a, b) for s, a, b in rows if s in (23, 29)])
            print("   even s with surplus>0:", [s for s, a, b in rows if s % 2 == 0 and a > b])
            print("   table:", [(s, a, b) for s, a, b in rows if s in (1, 3, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21)])
