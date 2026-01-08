"""
Modular solver for 1D Kronig-Penney using plane-wave expansion.
- Only use numpy to convert assembled matrix to an ndarray and call eigvalsh for eigenvalues.
"""
import math
import cmath
import numpy as np
from numpy.linalg import eigvalsh
import matplotlib.pyplot as plt

# ---------------------------
# Physical and model helpers
# ---------------------------
def constant():
    hbar = 1.054571817e-34
    m_e = 9.10938356e-31
    eV = 1.602176634e-19
    hbar2_2m_SI = hbar**2 / (2.0 * m_e)
    return hbar2_2m_SI * 1e18 / eV  # eV * nm^2

# ---------------------------
# 手写 FFT 实现
# ---------------------------
def fft_recursive(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])
    factor = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + factor[k] for k in range(N // 2)] + \
           [even[k] - factor[k] for k in range(N // 2)]

# ---------------------------
# Potential and Fourier
# ---------------------------
def make_kp_potential_one_cell(a, Lw, Ub, N_x):
    dx = a / N_x
    x_list = [i * dx for i in range(N_x)]
    V_list = []
    for xv in x_list:
        if (xv >= Lw) and (xv < a):
            V_list.append(Ub)
        else:
            V_list.append(0.0)
    return x_list, V_list

def compute_V_G(V_list, x_list, a, G_list):
    """
    使用手写FFT计算V_G:
      1. 对V(x)做FFT得到傅里叶系数
      2. 对齐G_list的顺序输出
    """
    N = len(V_list)
    fft_res = fft_recursive(V_list)
    # normalize并移动0频到中间（类似fftshift）
    fft_res = [v / N for v in fft_res]
    half = N // 2
    fft_shifted = fft_res[half:] + fft_res[:half]
    # 取与G_list对应的部分
    V_G = []
    stepG = 2 * math.pi / a
    for G in G_list:
        m = int(round(G / stepG))
        idx = (m + N // 2) % N
        V_G.append(fft_shifted[idx])
    return V_G

def assemble_V_matrix_from_VG(m_vals, V_G_all):
    m_to_idx = {m: idx for idx, m in enumerate(m_vals)}
    dim = len(m_vals)
    Vmat = [[0+0j for _ in range(dim)] for __ in range(dim)]
    for i, mi in enumerate(m_vals):
        for j, mj in enumerate(m_vals):
            diff = mi - mj
            if diff in m_to_idx:
                Vmat[i][j] = V_G_all[m_to_idx[diff]]
            else:
                Vmat[i][j] = 0+0j
    return Vmat

# ---------------------------
# Hamiltonian assembly & eigen
# ---------------------------
def build_H_matrix(k, G_vals, V_matrix, prefactor):
    dim = len(G_vals)
    H_py = [[0+0j for _ in range(dim)] for __ in range(dim)]
    for i, G in enumerate(G_vals):
        kinetic = prefactor * (k + G)**2
        H_py[i][i] = kinetic + V_matrix[i][i]
    for i in range(dim):
        for j in range(dim):
            if i != j:
                H_py[i][j] = V_matrix[i][j]
    H_np = np.array(H_py, dtype=np.complex128)
    H_np = 0.5 * (H_np + H_np.conj().T)
    return H_np

def compute_bands(k_list, G_vals, V_matrix, prefactor, n_bands):
    n_k = len(k_list)
    bands = [[0.0 for _ in range(n_k)] for __ in range(n_bands)]
    for ik, k in enumerate(k_list):
        Hk = build_H_matrix(k, G_vals, V_matrix, prefactor)
        eigs = eigvalsh(Hk)
        for nb in range(n_bands):
            bands[nb][ik] = float(eigs[nb].real)
    return bands

# ---------------------------
# Utility plotting / main run
# ---------------------------
def plot_bands(k_list, bands, title="Band structure"):
    plt.figure(figsize=(8,5))
    for band in bands:
        plt.plot(k_list, band)
    plt.xlabel("k (1/nm)")
    plt.ylabel("Energy (eV)")
    plt.xlim(k_list[0],k_list[-1])
    plt.ylim(0,14)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./band.png',dpi=300)

def run_kp_pw_example():
    """
    main workflow
    """
    U0 = 2.0
    L_W = 0.9
    L_B = 0.1
    a = L_W + L_B
    N_x = 1024    # number of x sample-points 
    N_G = 25      # number of trunctated PW basis      
    N_bands = 6   # number of bands to compute
    b = 2.0 * math.pi / a  # reciprocal vector
    n_k = 201     # number of kpts to compute in 1BZ

    x_list, V_list = make_kp_potential_one_cell(a, L_W, U0, N_x)
    m_vals = list(range(-N_G, N_G+1))
    G_vals = [2.0 * math.pi * m / a for m in m_vals]

    V_G_all = compute_V_G(V_list, x_list, a, G_vals)
    V_matrix = assemble_V_matrix_from_VG(m_vals, V_G_all)

    prefactor = constant()
    k_list = [(-b/2.0) + i * (b / (n_k - 1)) for i in range(n_k)]

    bands = compute_bands(k_list, G_vals, V_matrix, prefactor, N_bands)
    H_k0 = build_H_matrix(0.0, G_vals, V_matrix, prefactor)
    eigs_k0 = eigvalsh(H_k0)
    print("\nFirst 10 eigenvalues at k=0 (eV):")
    for i in range(min(10, len(eigs_k0))):
        print(f"  {i+1:2d}: {float(eigs_k0[i]):.6f}")
    print("\nRequested: first 3 eigenvalues at k=0:")
    for i in range(3):
        print(f"  E_{i+1} = {float(eigs_k0[i]):.6f} eV")
    plot_bands(k_list, bands, title="Band structure of KP potential (PW expansion)")

if __name__ == "__main__":
    run_kp_pw_example()
