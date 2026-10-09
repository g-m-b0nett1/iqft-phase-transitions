"""Independent check of Sec. 1.5.2 (Ising chain in an imaginary field) and Fig. 1.6.
Zeroes of Z_N = lambda_+^N + lambda_-^N computed from the eigenvalue formula (1.5.6),
not from the polynomial, and compared with the TikZ coordinates in the figure."""
import numpy as np, mpmath as mp
mp.mp.dps = 30
K, N = mp.mpf('0.5'), 24
def Z(phi):
    c = mp.e**K*mp.cos(phi); s = mp.sqrt(mp.e**(-2*K) - mp.e**(2*K)*mp.sin(phi)**2)
    return (c+s)**N + (c-s)**N
phie = mp.asin(mp.e**(-2*K))
# zeroes: N alpha = pi(k+1/2), alpha from (1.5.8), phi in (phi_e, pi - phi_e)
def alpha(phi):
    return mp.atan2(mp.sqrt(mp.e**(2*K)*mp.sin(phi)**2 - mp.e**(-2*K)), mp.e**K*mp.cos(phi))
zs = []
for k in range(N):
    target = mp.pi*(k+mp.mpf(1)/2)/N
    if target >= mp.pi: break
    lo, hi = phie, mp.pi/2*2-phie
    for _ in range(200):
        mid=(lo+hi)/2
        if alpha(mid) < target: lo = mid
        else: hi = mid
    zs.append(lo)
print("2 phi_e (deg) =", float(2*phie*180/mp.pi))
print("number of zeroes with 0<phi<pi:", len(zs))
pts = sorted([(float(mp.cos(2*p)), float(mp.sin(2*p))) for p in zs])
print(" ".join(f"({x:.4f},{y:.4f})" for x, y in pts))
print("max |Z| at the computed zeroes (relative):", max(float(abs(Z(p))/ (mp.e**K)**N) for p in zs))
# edge-density formula check (1.5.11): numerical d alpha/d phi near edge vs asymptotic
for d in [1e-3, 1e-5]:
    p = phie + d
    num = mp.diff(alpha, p)/mp.pi
    asym = (1/mp.pi)*mp.sqrt(mp.tan(phie)/2)*d**-0.5
    print("delta", d, "g num/asym =", float(num/asym))
