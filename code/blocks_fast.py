"""Block multiplicities n_x of phi_ab via a discrete sine transform:
   n_x = 2 sum_s sin(pi s x/h) cot(pi s/h) psi_a^s psi_b^s   (s over exponents, with multiplicity),
from sinh((h-x)y)/sinh(hy) = (1/h) sum_{s=1}^{h-1} sin(pi s/h) sin(pi s x/h)/(cosh y - cos(pi s/h)).
Checks integrality (|n - round(n)| small) and positivity, x = 1..h-1.
"""
import numpy as np

def inc(kind, l):
    I = np.zeros((l, l))
    def link(i, j): I[i, j] = I[j, i] = 1
    for i in range((l - 1) if kind == 'A' else (l - 2)): link(i, i + 1)
    if kind == 'D': link(l - 3, l - 1); h = 2 * l - 2
    elif kind == 'E': link(2, l - 1); h = {6: 12, 7: 18, 8: 30}[l]
    else: h = l + 1
    return I, h

def mult(kind, l):
    I, h = inc(kind, l)
    ev, V = np.linalg.eigh(I)
    s = np.arccos(np.clip(ev / 2, -1, 1)) * h / np.pi
    x = np.arange(1, h)
    S = np.sin(np.pi * np.outer(x, s) / h) * (2 / np.tan(np.pi * s / h))  # (h-1) x l
    N = np.einsum('xk,ak,bk->abx', S, V, V)
    return N, h

# validation against the exact sympy output for E8, S_77 = (2)(10)(12)(18)(20)(28) and A4 S_22 = (2)^2(4)
N, h = mult('E', 8); print("E8 S_77:", {x + 1: int(round(v)) for x, v in enumerate(N[6, 6]) if round(v)})
N, h = mult('A', 4); print("A4 S_22:", {x + 1: int(round(v)) for x, v in enumerate(N[1, 1]) if round(v)})
worst_int, worst_neg, cnt = 0, 0, 0
for kind, rng in [('A', range(1, 81)), ('D', range(4, 81)), ('E', [6, 7, 8])]:
    for l in rng:
        N, h = mult(kind, l)
        R = np.round(N)
        worst_int = max(worst_int, np.max(np.abs(N - R)))
        worst_neg = min(worst_neg, np.min(R))
        nonzero = np.all(np.sum(R, axis=2)[np.triu_indices(l)] > 0) if l > 1 else True
        if kind != 'A' or l > 1:
            assert nonzero, (kind, l)
        cnt += 1
print(f"{cnt} diagrams (A1-A80, D4-D80, E6-E8): max |n - integer| = {worst_int:.1e}; min n_x = {int(worst_neg)}; every S_ab (except A1) has at least one block")
