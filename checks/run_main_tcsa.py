"""Runs the main code/tcsa_ising.py at the N_cut given on the command line (independent check of Sec. 3.4)."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CODE = _os.path.join(_HERE, '..', 'code')
import sys, time, numpy as np
sys.path.insert(0, _CODE)
import tcsa_ising as T
for Nc in [int(a) for a in sys.argv[1:]]:
    t=time.time(); H0,B = T.build(Nc)
    rc = T.collision(H0,B,lo=2.0,hi=2.8,tol=1e-9)
    out=[f"Ncut={Nc} states={len(H0)} rc={rc:.7f}"]
    for r in [10,20]:
        gR=(r/(2*np.pi*T.FATEEV))**(15/8); e=np.sort(T.spectrum(H0,B,gR,8).real)
        out.append(f" r={r}: RE0/r^2={(2*np.pi*e[0])/r**2:.6f} gaps/m1="+" ".join(f"{(e[i]-e[0])*2*np.pi/r:.4f}" for i in range(1,5)))
    # second order ratio
    for r in [0.05,0.1,0.2]:
        gR=(r/(2*np.pi*T.FATEEV))**(15/8); e=np.sort(T.spectrum(H0,B,gR,2).real)
        from mpmath import gamma
        X=1/8; g=lambda x: gamma(x)/gamma(1-x); I2=float(np.pi*g(X/2)**2*g(1-X))
        out.append(f" r={r}: shift/(-pi I2 gR^2) = {(e[0]+1/24)/(-np.pi*I2*gR**2):.6f}")
    print("\n".join(out), f"[{time.time()-t:.0f}s]", flush=True)
