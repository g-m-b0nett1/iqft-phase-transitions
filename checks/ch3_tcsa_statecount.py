"""Check of the state counts of the TCSA (Sec. 3.4): exact numbers of spin-zero states with
descendant level n = nbar <= Ncut/2, from the Ising characters, against the ranks retained
by the main code (Gram-matrix eigenvalue threshold) at each level."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CODE = _os.path.join(_HERE, '..', 'code')
import sys, numpy as np
sys.path.insert(0, _CODE); sys.path.insert(0, _HERE)
import ch4_counting_independent as C
chi0, chie, chis = C.ising_chars()
exact = {Nc: sum(chi0[n]**2 + chie[n]**2 + chis[n]**2 for n in range(Nc // 2 + 1)) for Nc in (8, 12, 16, 24, 32)}
print("exact numbers of states:", exact)
import tcsa_ising as T
for lab, ch in (("1", chi0), ("e", chie), ("s", chis)):
    M = T.Module(lab, 16)
    kept = [M.dim(n) for n in range(17)]
    print(lab, "kept by code :", kept)
    print(lab, "character    :", ch[:17])
