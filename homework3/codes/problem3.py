# Gaussian basis: phi_i(x) = sqrt(v_i/pi) * exp(-v_i * (x - s_i)**2)
# Units: Hartree atomic units (ħ = m = e = 1)
# v_i, v_j denote widths; s_i, s_j denote centers.

import numpy as np

def overlap_S(vi, si, vj, sj):
    """S_ij = ∫ phi_i(x) phi_j(x) dx (analytical)."""
    a, b = vi, vj
    cfac = np.sqrt(a * b) / np.sqrt(np.pi)
    p = a + b
    d = si - sj
    expo = np.exp(-a * b * d * d / p)
    return cfac / np.sqrt(p) * expo

def kinetic_T(vi, si, vj, sj):
    """
    T_ij = <phi_i| -1/2 d^2/dx^2 |phi_j>
         = 1/2 ∫ phi_i'(x) phi_j'(x) dx (analytical form used below):
    T_ij = S_ij * (a*b/p) * (1 - 2ab*d^2/p), where a=vi, b=vj, p=a+b, d=si-sj.
    """
    a, b = vi, vj
    p = a + b
    d = si - sj
    Sij = overlap_S(a, si, b, sj)
    return Sij * (a * b / p) * (1.0 - 2.0 * a * b * d * d / p)

def _common_I0_mu_p(vi, si, vj, sj):
    """
    Helper for polynomial potentials:
      I0 = ∫ exp[-a(x-si)^2 - b(x-sj)^2] dx
      mu = (a*si + b*sj) / (a + b)
      p  = a + b
    Note: S_ij = sqrt(vi/pi)*sqrt(vj/pi) * I0
    """
    a, b = vi, vj
    p = a + b
    mu = (a * si + b * sj) / p
    d = si - sj
    I0 = np.sqrt(np.pi / p) * np.exp(-a * b * d * d / p)
    return I0, mu, p

def V_x2_element(vi, si, vj, sj):
    """Potential matrix element for V(x) = x^2."""
    _, mu, p = _common_I0_mu_p(vi, si, vj, sj)
    Sij = overlap_S(vi, si, vj, sj)
    # ∫ x^2 e^{-p(x-mu)^2} dx = (mu^2 + 1/(2p)) * sqrt(pi/p)
    moment2 = mu * mu + 1.0 / (2.0 * p)
    return Sij * moment2

def V_x4_minus_x2_element(vi, si, vj, sj):
    """Potential matrix element for V(x) = x^4 - x^2."""
    _, mu, p = _common_I0_mu_p(vi, si, vj, sj)
    Sij = overlap_S(vi, si, vj, sj)
    # ∫ x^4 e^{-p(x-mu)^2} dx = (mu^4 + 3 mu^2/p + 3/(4 p^2)) * sqrt(pi/p)
    moment2 = mu * mu + 1.0 / (2.0 * p)
    moment4 = mu**4 + 3.0 * mu * mu / p + 3.0 / (4.0 * p * p)
    return Sij * (moment4 - moment2)

def assemble_matrices(nus, centers, potential="x2"):
    """Assemble Hamiltonian H and overlap S."""
    n = len(nus)
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            si, sj = centers[i], centers[j]
            nui, nuj = nus[i], nus[j]
            Sij = overlap_S(nui, si, nuj, sj)
            Tij = kinetic_T(nui, si, nuj, sj)
            if potential == "x2":
                Vij = V_x2_element(nui, si, nuj, sj)
            elif potential == "x4-x2":
                Vij = V_x4_minus_x2_element(nui, si, nuj, sj)
            else:
                raise ValueError("Unknown potential")
            S[i, j] = S[j, i] = Sij
            T[i, j] = T[j, i] = Tij
            V[i, j] = V[j, i] = Vij
    H = T + V
    return H, S

def generalized_eig(H, S):
    """
    Solve the generalized eigenproblem H c = E S c ,E is an eigenvalue.
    Use Cholesky S = R^T R and transform to a standard symmetric eigenproblem:
        A = R^{-T} H R^{-1},  A y = E y,  y = Rc.
    Avoid explicit inverses by solving triangular systems.

    here in np.linalg.cholesky(S) it decomposes S = L@L^T  
    and only return the lower triangluar matrix L               
    so the codes look a bit different from the above expression: because  R -> L^T
    """
    L = np.linalg.cholesky(S)               # S = L L^T,the output is of cholesky is L lower-triangular  L=R^T
    X = np.linalg.solve(L, H)               # X = L^{-1} H
    A = np.linalg.solve(L, X.T)           
    # Symmetric eigen-decomposition
    E, Y = np.linalg.eigh(A)                # E ascending, Y columns are eigenvectors
    # Recover coefficients in the original (non-orthonormal) basis: c = R^{-1} y
    C = np.linalg.solve(L.T, Y)
    return E, C

def fixed_centers_basis(centers, v):
    """Build basis with fixed centers and a shared width v."""
    n = len(centers)
    vs = np.full(n, float(v))
    return vs, np.asarray(centers, dtype=float)

def scan_opt_nu_for_fixed_centers(centers, potential="x2", nu_grid=None, index=None):
    """
    Scan a grid of widths v for fixed centers and pick the best by minimizing E[index].
    the fixed center basis has the same width here.
    """
    if nu_grid is None:
        nu_grid = np.linspace(0.2, 3.0, 60)
    best = None
    for nu in nu_grid:
        nus, cs = fixed_centers_basis(centers, nu)
        H, S = assemble_matrices(nus, cs, potential)
        E, _ = generalized_eig(H, S)
        score = E[index]
        if best is None or score < best[0]:
            best = (score, nu)
    return best

def solve_and_report(nus, centers, potential,index,v_best):
    """Assemble matrices, solve, and print the lowest three energies."""
    H, S = assemble_matrices(nus, centers, potential)
    E, C = generalized_eig(H, S)
    print(f"best width = {v_best:.4f} ", "energy",index+1,f" ={E[index]:.8f}", )
    return E, C

def main():
    np.set_printoptions(precision=6, suppress=True)

    # Fixed centers on [-3, 3]; scan a shared width v to minimize the ground energy.
    centers = np.linspace(-3.0, 3.0, 9)

    for pot in ("x2", "x4-x2"):
        print(f"[Fixed centers] potential = {pot} "," basis size =",len(centers))
        for i in [0,1,2]:
            bestE0, v_best = scan_opt_nu_for_fixed_centers(centers, potential=pot,index=i)
            nus, cs = fixed_centers_basis(centers, v_best)
            solve_and_report(nus, cs, pot, index=i,v_best=v_best)

    # Reference for V=x^2: in these units V = (1/2) ω^2 x^2 with ω = sqrt(2)
    #w = np.sqrt(2.0)
    #exact = [(n + 0.5) * w for n in range(3)]
    #print("\n[Reference] Harmonic oscillator exact (V=x^2):"," , ".join(f"{e:.8f}" for e in exact))

if __name__ == "__main__":
    main()