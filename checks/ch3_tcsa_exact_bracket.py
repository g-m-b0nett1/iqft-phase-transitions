"""Exact-basis TCSA at large N_cut without a full bisection: checks that the two lowest levels at imaginary
field are real at r = r_lo and a complex pair at r = r_hi, and prints the bulk energy and gaps at r = 10, 20.
Usage: python3 ch3_tcsa_exact_bracket.py Ncut r_lo r_hi"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, _HERE)
import sys, time, numpy as np
import ch3_tcsa_exact as E, ch3_tcsa_independent as R
Nc, rlo, rhi = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
t = time.time(); H0, V = E.build(Nc)
print(f"Ncut={Nc}: {len(H0)} states (exact count); built in {time.time()-t:.0f} s", flush=True)
for r in (rlo, rhi):
    e = R.levels(H0, V, 1j * (r / (2 * np.pi * R.KAPPA)) ** (15 / 8))
    print(f"  r={r}: two lowest levels {e[0]:.8f}, {e[1]:.8f}  -> {'complex pair' if abs(e[0].imag) > 1e-10 else 'real'}", flush=True)
for r in (10, 20):
    e = R.levels(H0, V, (r / (2 * np.pi * R.KAPPA)) ** (15 / 8)).real
    print(f"  r={r}: 2 pi e0/r^2 = {2*np.pi*e[0]/r**2:.6f}; gaps/m1 = " + " ".join(f"{(e[i]-e[0])*2*np.pi/r:.4f}" for i in (1, 2, 3)), flush=True)
