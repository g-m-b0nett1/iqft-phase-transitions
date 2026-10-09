# Phase transitions in integrable QFTs: code

Python code accompanying *Phase transitions in integrable QFTs*, a project I'm working on that revises and extends my master's thesis (Utrecht University, 2025). It computes the numbers, tables and figures used in the text and checks its exact statements numerically. References to chapters, sections, remarks and equations are to that work.

## Layout

- `code/`: the main scripts.
- `checks/`: independent recomputations of results in `code/`, written separately to test them;
  `checks/outputs/` holds the outputs they produced.
- `generated/`: LaTeX fragments (numbers, tables, TikZ coordinates) written by the scripts in `code/`.
- `figures/`: figures written by the scripts in `code/`.

Files in `generated/` and `figures/` are never edited by hand: change the script and rerun it.

## Setup

Python 3.12 with the packages in `requirements.txt`:

    pip install -r requirements.txt

Run every script from the repository root, e.g. `python3 code/smatrix_ade.py`.

## Main scripts (`code/`)

| Script | Topic | Notes |
|---|---|---|
| `ising1d_zeroes.py` | Lee–Yang zeroes of the periodic one-dimensional Ising chain (Ch. 1) | `--tex` writes `generated/ising1d_zeroes.tex` |
| `yanglee_mass.py` | Coupling–mass relation of the scaling Yang–Lee model, from Al. B. Zamolodchikov (1995) (Ch. 3) | `--tex` writes `generated/yanglee_mass.tex` |
| `tcsa_ising.py` | Truncated conformal space approach for the Ising model perturbed by the spin field: E8 mass ratios, bulk energy, second-order check, finite-size Yang–Lee edge r_c at imaginary field (Ch. 3) | `python3 code/tcsa_ising.py [Ncut]`; `--tex` writes `generated/tcsa_ising.tex` (about 4 min) |
| `relative_bound.py` | Relative boundedness (Kato) of the perturbation with respect to H0, Ising with σ and ε (Ch. 3) | about 9 min |
| `relative_bound_minimal.py` | The same norms for M(4,5), M(6,7), M(5,6) (Ch. 3) | `python3 code/relative_bound_minimal.py tim\|tcp\|m56 [Ncut ...]` |
| `counting.py` | Zamolodchikov's counting argument for conserved charges (Ch. 4) | |
| `galois_orbits.py` | Galois orbits of the exponents and masses of E6, E7, E8 (Ch. 4) | |
| `smatrix_ade.py` | Minimal ADE S-matrices from the inverse q-deformed Cartan matrix: exact block content, positivity of the kernels, unitarity, crossing, bound states and bootstrap equations (Ch. 5) | `--tex --fig` writes the E8 tables and `figures/kernels_e8.pdf` (about 10 min) |
| `ade_kernels.py`, `blocks_fast.py` | Block decomposition and sign of the ADE TBA kernels: exact to rank 10, numerically integer-exact to rank 80 (Ch. 5) | |
| `cluster_signs.py` | Signs of the cluster coefficients of the ground-state TBA branch | |

## Independent checks (`checks/`)

| Script | Checks |
|---|---|
| `ch1_ising_chain.py` | Ising chain zeroes from the eigenvalue formula, against `ising1d_zeroes.py` |
| `ch3_constants.py` | Constants of Ch. 3 (E8 mass–coupling constant, integrals, Yang–Lee constants) |
| `ch3_free_fermion_radius.py` | Radius of convergence for the free Majorana fermion |
| `ch3_tcsa_independent.py` | TCSA for Ising + σ written from scratch (does not import `code/`) |
| `ch3_tcsa_exact.py`, `ch3_tcsa_exact_bracket.py` | TCSA in an exactly orthonormalised basis (exact rank, 60-digit Cholesky); about 4 min at Ncut = 24, 30 min at 28, 40 min for the bracket at 32 |
| `ch3_tcsa_statecount.py` | State counts of the main TCSA basis against the Ising characters |
| `ch3_relbound_exact.py`, `ch3_relbound_basischeck.py`, `ch3_chiral_estimate_check.py` | Relative-bound norms in the exact basis |
| `ch4_counting_independent.py` | Counting tables from independently written characters |
| `ch4_galois_independent.py` | Galois orbits from the action of (Z/2hZ)* on roots of unity, without minimal polynomials |
| `run_main_tcsa.py` | Runs `code/tcsa_ising.py` at given Ncut values for comparison |

## Known limitation

`code/tcsa_ising.py` (and `relative_bound*.py`, which use it) orthonormalises the Virasoro modules with a floating-point threshold on the Gram matrix. From descendant level 11 on (level 8–10 for the other minimal models) this discards genuine states (`checks/outputs/tcsa_statecount.txt`). The exact-basis recomputation in `checks/ch3_tcsa_exact.py` keeps all states and gives the same results to the digits quoted: r_c = 2.3778243 at Ncut = 24, 28 and 32 (`checks/outputs/exact_tcsa*.txt`, `main_tcsa.txt`).

## Licence

All rights reserved (see `LICENSE`). The code is public so that it can be read; it is work in progress,
and no permission to use, copy, modify or distribute it is granted.
