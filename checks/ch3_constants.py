"""Independent recomputation of the constants of Chapter 3."""
import mpmath as mp
mp.mp.dps = 30
G = mp.gamma
# Fateev's E8 mass-coupling constant, eq. (3.1.4)
k = 4*mp.sin(mp.pi/5)*G(mp.mpf(1)/5)/(G(mp.mpf(2)/3)*G(mp.mpf(8)/15)) * \
    (4*mp.pi**2*G(mp.mpf(3)/4)*G(mp.mpf(13)/16)**2/(G(mp.mpf(1)/4)*G(mp.mpf(3)/16)**2))**(mp.mpf(4)/15)
print("kappa_E8 =", k)
# I2 for X = 1/8, eq. (3.3.4), and a direct numerical integral as an independent check
gam = lambda x: G(x)/G(1-x)
X = mp.mpf(1)/8
I2 = mp.pi*gam(X/2)**2*gam(1-X)
print("I2(1/8) closed form =", I2)
def I2num(X):
    # polar coordinates; integrand |z|^{X-2}|1-z|^{-2X}; inner region rho<1 with rho = u^{1/X}
    # (removes the rho^{X-1} singularity), outer region rho>1 mapped by rho -> 1/rho
    ang = lambda rho: mp.quad(lambda th: abs(1-rho*mp.e**(1j*th))**(-2*X), [0, mp.mpf(1)/64, mp.mpf(1)/8, mp.pi])
    inner = mp.quad(lambda u: ang(u**(1/X))/X, [0, mp.mpf(1)/2, mp.mpf(9)/10, 1])
    # outer: rho = 1/t, d rho = dt/t^2, rho^{X-1} = t^{1-X}; |1 - e^{i th}/t|^{-2X} = t^{2X}|t - e^{i th}|^{-2X}
    outer = mp.quad(lambda t: t**(X-1)*mp.quad(lambda th: abs(t-mp.e**(1j*th))**(-2*X), [0, mp.mpf(1)/64, mp.mpf(1)/8, mp.pi]), [0, mp.mpf(1)/2, mp.mpf(9)/10, 1])
    outer2 = mp.quad(lambda u: mp.quad(lambda th: abs(u**(1/X)-mp.e**(1j*th))**(-2*X), [0, mp.mpf(1)/64, mp.mpf(1)/8, mp.pi])/X, [0, mp.mpf(1)/2, mp.mpf(9)/10, 1])
    return 2*(inner + outer2)
mp.mp.dps = 15
print("I2(1/8) numerical    =", I2num(mp.mpf(1)/8))
print("I2(0.3) closed/num   =", mp.pi*gam(0.15)**2*gam(0.7), I2num(mp.mpf('0.3')))
mp.mp.dps = 30
# Yang-Lee mass constant: g = 0.0970484562986085 i m^{12/5}  ->  m = (|g|/0.0970...)^{5/12}
print("YL mass constant =", (1/mp.mpf('0.0970484562986085'))**(mp.mpf(5)/12))
# two-level estimate of r_c
print("two-level r_c =", 2*mp.pi*k*(32*mp.pi)**(-mp.mpf(8)/15))
# E8 masses
m = [1, 2*mp.cos(mp.pi/5), 2*mp.cos(mp.pi/30), 4*mp.cos(mp.pi/5)*mp.cos(7*mp.pi/30),
     4*mp.cos(mp.pi/5)*mp.cos(2*mp.pi/15), 4*mp.cos(mp.pi/5)*mp.cos(mp.pi/30),
     8*mp.cos(mp.pi/5)**2*mp.cos(7*mp.pi/30), 8*mp.cos(mp.pi/5)**2*mp.cos(2*mp.pi/15)]
print("E8 masses:", [mp.nstr(x, 8) for x in m])
# E8 bulk energy: -m1^2 / (8 (sin(2pi/3) + sin(2pi/5) + sin(pi/15)))  (closed form as recalled;
# it reproduces the value -0.0617286 quoted from Fateev 1994 / Fonseca-Zamolodchikov 2003)
eps = -1/(8*(mp.sin(2*mp.pi/3)+mp.sin(2*mp.pi/5)+mp.sin(mp.pi/15)))
print("eps_bulk/m1^2 =", eps)
# Yang-Lee structure constant |C| = Gamma(1/5)^{3/2} Gamma(2/5)^{1/2} / (5 Gamma(4/5)^{3/2} Gamma(3/5)^{1/2})
Cyl = G(mp.mpf(1)/5)**1.5*G(mp.mpf(2)/5)**0.5/(5*G(mp.mpf(4)/5)**1.5*G(mp.mpf(3)/5)**0.5)
print("|C_YL| =", Cyl)
# |r_c| implied by Al. Zamolodchikov's G0 = -14.3 +- 0.4 with G = r^{12/5}
for G0 in (13.9, 14.3, 14.7): print("G0", G0, "-> |r_c| =", mp.mpf(G0)**(mp.mpf(5)/12))
