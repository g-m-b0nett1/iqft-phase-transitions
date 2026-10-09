"""Free Majorana fermion, antiperiodic sector (Sec. 3.3): Taylor coefficients in u = r^2 of the
convergent part -1/2 sum_k (omega_k - |p_k| - m^2/(2|p_k|)), R = 1, p_k = 2 pi (k+1/2).
c_n = -binom(1/2, n) * 2 * sum_{k>=0} p_k^{1-2n}  (n >= 2);  ratio test -> radius pi^2."""
import mpmath as mp
mp.mp.dps = 40
def c(n):
    s = (2*mp.pi)**(1-2*n) * (2**(2*n-1) - 1) * mp.zeta(2*n-1)   # sum_{k>=0} (2pi(k+1/2))^{1-2n}
    return -mp.binomial(mp.mpf(1)/2, n) * s     # -1/2 * 2 (both signs of k)
for n in (5, 10, 20, 40):
    print(n, "c_n/c_{n+1} =", mp.nstr(c(n)/c(n+1), 12), "  -pi^2 =", mp.nstr(-mp.pi**2, 12))
# direct check of the series against brute-force summation at u = 2
u = mp.mpf(2)
brute = -mp.mpf(1)/2*2*mp.nsum(lambda k: mp.sqrt(u + (2*mp.pi*(k+0.5))**2) - 2*mp.pi*(k+0.5) - u/(2*2*mp.pi*(k+0.5)), [0, mp.inf])
series = mp.fsum(c(n)*u**n for n in range(2, 80))
print("brute =", mp.nstr(brute, 15), " series =", mp.nstr(series, 15))
