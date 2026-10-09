"""Taylor coefficients of Lambda_0(z) for one fugacity, by formal iteration of the TBA
    eps_a = r m_a cosh(th) - sum_b (phi_ab/2pi) * log(1 + z e^{-eps_b}),
    Lambda_0 = sum_a int dth/(2pi) r m_a cosh(th) log(1 + z e^{-eps_a})   (dimensionless, per unit L/R)
Substituting z = -x, l_a := -log(1 - x e^{-eps_a}); claim: all x-coefficients of l_a >= 0 when phi <= 0.
Checks (-1)^{N-1} b_N >= 0 and estimates the radius against the free-fermion bound e^{r m_1}.
"""
import numpy as np, sys

def series_exp(c, K):     # exp of power series with c[0]=0 ; c: list of arrays
    out = [np.ones_like(c[1])] + [np.zeros_like(c[1]) for _ in range(K)]
    for n in range(1, K + 1):  # f' = c' f
        out[n] = sum(k * c[k] * out[n - k] for k in range(1, n + 1)) / n
    return out

def run(name, masses, phi, r, K=14, T=12.0, M=1601):
    th = np.linspace(-T, T, M); d = th[1] - th[0]
    n = len(masses)
    # kernel matrices on grid differences
    D = th[:, None] - th[None, :]
    Kab = [[phi(a, b, D) / (2 * np.pi) * d for b in range(n)] for a in range(n)]
    nu = [r * masses[a] * np.cosh(th) for a in range(n)]
    # l[a][k]: coefficient of x^k (k>=1) of l_a(th)
    l = [[np.zeros(M) for _ in range(K + 1)] for _ in range(n)]
    for N in range(1, K + 1):
        # e^{-eps_a} = e^{-nu_a} exp( sum_b psi_ab * l_b ), psi = -phi
        for a in range(n):
            g = [np.zeros(M)] + [sum(-Kab[a][b] @ l[b][k] for b in range(n)) for k in range(1, K + 1)]
            E = series_exp(g, K)
            u = [np.zeros(M)] + [np.exp(-nu[a]) * E[k - 1] for k in range(1, K + 1)]  # u = x e^{-eps}
            # l = sum_j u^j / j, only need order N
            pw = u[:]; acc = u[N].copy()
            for j in range(2, N + 1):
                new = [np.zeros(M)] * (K + 1)
                new = [sum(pw[i] * u[k - i] for i in range(1, k)) if k >= 2 else np.zeros(M) for k in range(K + 1)]
                pw = new; acc = acc + pw[N] / j
            l[a][N] = acc
    # coefficients of -Lambda_0(-x)
    c = [sum(np.sum(r * masses[a] * np.cosh(th) * l[a][k]) * d / (2 * np.pi) for a in range(n)) for k in range(1, K + 1)]
    bN = [(-1) ** (k - 1) * ck for k, ck in zip(range(1, K + 1), c)]  # b_N = coefficient of z^N
    mins = min(np.min(l[a][k]) for a in range(n) for k in range(1, K + 1))
    ratios = [c[k] / c[k + 1] for k in range(K - 4, K - 1)]
    print(f"{name}, r={r}: min over theta,N of l-coefficients = {mins:.2e}; "
          f"(-1)^(N-1) b_N > 0 for all N<={K}: {all(ci > 0 for ci in c)}; "
          f"ratio c_N/c_N+1 (last) = {', '.join(f'{x:.4f}' for x in ratios)}; free-fermion bound e^(r m1) = {np.exp(r*min(masses)):.4f}")

# scaling Yang-Lee: phi = -4 sqrt3 cosh / (1 + 2 cosh 2th)
yl = lambda a, b, D: -4 * np.sqrt(3) * np.cosh(D) / (1 + 2 * np.cosh(2 * D))
# A2 (3-state Potts), equal masses: S_11 = S_22 = (2), S_12 = (1), h = 3
blk = lambda x, h, D: -np.sin(np.pi * x / h) / (np.cosh(D) - np.cos(np.pi * x / h))
a2 = lambda a, b, D: blk(2, 3, D) if a == b else blk(1, 3, D)
free = lambda a, b, D: 0 * D
for r in [0.5, 1.0, 2.0]:
    run("free fermion", [1.0], free, r)
    run("scaling Yang-Lee", [1.0], yl, r)
    run("A2 (Potts)", [1.0, 1.0], a2, r)
