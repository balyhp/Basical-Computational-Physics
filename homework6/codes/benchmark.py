"""
Kronig-Penney (1D) — plane-wave + FFT implementation
Computes lowest few bands E_n(k) using plane-wave basis and FFT for V_G.
"""
import numpy as np
from numpy.fft import fft, fftshift, ifftshift
from scipy.linalg import eigh

# ---------- Physical constants ----------
hbar = 1.054571817e-34    # J s
m_e = 9.1093837015e-31    # kg
eV = 1.602176634e-19      # J

# ---------- Problem parameters (user) ----------
U0_eV = 2.0               # barrier height in eV
L_B_nm = 0.1              # barrier width nm
L_W_nm = 0.9              # well (zero) width nm
a_nm = 1.0                # period nm

# convert to SI (meters, Joules)
U0 = U0_eV * eV
L_B = L_B_nm * 1e-9
L_W = L_W_nm * 1e-9
a = a_nm * 1e-9

# ---------- numerical parameters ----------
Npw = 401                 # number of plane waves (odd recommended)
Ngrid = Npw               # number of real-space sample points per period (use Ngrid=Npw so FFT frequencies align)
# choose Npw ~ 101-401 for convergence tests

# ---------- build real-space grid and potential V(x) ----------
x = np.linspace(0, a, Ngrid, endpoint=False)  # sample in [0,a)
# define periodic Kronig-Penney potential: barrier of height U0 of width L_B, located e.g. at the right side of period
# we'll place barrier from x = 0 to x = L_B (any phase just shifts Fourier phases)
Vx = np.zeros_like(x)
# barrier region: 0 <= x < L_B
Vx[(x >= 0) & (x < L_B)] = U0

# ---------- compute FFT of V(x) to get discrete Fourier coefficients ----------
# discrete Fourier: V_G[n] corresponds to G = 2*pi * n / a with n = 0..Ngrid-1 (with fftshift we center negative frequencies)
V_G_raw = fft(Vx) / Ngrid    # unshifted (0..N-1)
V_G = fftshift(V_G_raw)     # center zero frequency at middle
# build corresponding G-grid (shifted)
n_indices = np.arange(-Ngrid//2, -Ngrid//2 + Ngrid)  # length Ngrid, gives ...,-2,-1,0,1,2,...
G_grid = 2.0 * np.pi * n_indices / a

# ---------- choose plane waves G list (centered) ----------
# choose Npw centered around G=0: n = -Npw//2 ... +Npw//2
half = Npw // 2
n_pw = np.arange(-half, half+1)   # length Npw (odd)
G_pw = 2.0 * np.pi * n_pw / a     # G values for plane-wave basis

# check that all G_pw are present in G_grid (they are if Ngrid = Npw and sampling aligned)
# find mapping from difference index (m = i-j) to V_G index:
# If we index V_G as V_G[k_index] with k_index = 0..Ngrid-1 mapping to n_indices above,
# then difference G_diff = G_pw[i]-G_pw[j] corresponds to integer (n_pw[i]-n_pw[j]) which maps into G_grid index = (n_pw[i]-n_pw[j]) + Ngrid//2
# We'll construct V_diff array accordingly.

# ---------- extract V_{m} for m = differences from -half..half ----------
# V_G_shifted array corresponds to G = 2pi * n_indices/a where n_indices = -Ngrid//2 ... +Ngrid//2-1
# create dict mapping integer n -> V_G value
n_to_V = {n_indices[i]: V_G[i] for i in range(Ngrid)}

# function to get V_G for integer n (may be zero if beyond sampled range)
def V_coeff(n):
    # due to aliasing, if n not in dict, we can take modulo Ngrid (periodic)
    # but with Ngrid == Npw this is fine for |n| <= half
    if n in n_to_V:
        return n_to_V[n]
    else:
        # alias (shouldn't normally be needed if configuration consistent)
        n_mod = ((n + Ngrid//2) % Ngrid) - Ngrid//2
        return n_to_V.get(n_mod, 0.0)

# ---------- function to build Hamiltonian at given Bloch k ----------
def build_H_k(k, Npw=Npw):
    """
    returns H (Npw x Npw Hermitian) for Bloch momentum k (in 1/m)
    """
    # Kinetic diagonal
    T = (hbar**2 / (2.0 * m_e)) * (k + G_pw)**2   # array length Npw
    H = np.zeros((Npw, Npw), dtype=np.complex128)
    # Fill diagonal kinetic
    for i in range(Npw):
        H[i, i] = T[i]
    # Fill potential via convolution: H_{i,j} += V_{G_i - G_j} = V_{2pi*(n_i-n_j)/a}
    for i in range(Npw):
        for j in range(Npw):
            dn = n_pw[i] - n_pw[j]    # integer difference index
            H[i, j] += V_coeff(dn)
    # H is Hermitian
    return H

# ---------- example: compute lowest 3 eigenvalues at k = 0 ----------
# choose k values (in 1/m), use reduced zone: k in [-pi/a, pi/a]
k_BZ = np.linspace(-np.pi / a, np.pi / a, 201)  # for band structure plotting if desired

# compute for a single k (e.g., k=0)
k0 = 0.0
H0 = build_H_k(k0, Npw=Npw)
# diagonalize (dense)
eigvals, eigvecs = eigh(H0)
# eigvals are in Joules; convert to eV for output
eig_eV = eigvals / eV
print("Lowest 6 eigenvalues at k=0 (eV):")
print(eig_eV[:6])

# ---------- compute bands E_n(k) for several k (optional) ----------
def compute_bands(k_array, num_bands=6):
    bands = np.zeros((len(k_array), num_bands))
    for idx, k in enumerate(k_array):
        H = build_H_k(k, Npw=Npw)
        w, _ = eigh(H)
        bands[idx, :] = (w[:num_bands] / eV)
    return bands

# example: compute first 3 bands vs k (this may take some time for many k and large Npw)
#bands3 = compute_bands(k_BZ, num_bands=3)
# now bands3 is (len(k_BZ),3) in eV

# ---------- notes ----------
# - Increase Npw until the first 3 eigenvalues converge (e.g., Npw=101,151,201 and compare).
# - Make sure Ngrid = Npw and sampling aligns so FFT coefficients map exactly to plane-wave G points.
# - The barrier phase (where barrier sits inside period) only shifts phases of V_G and changes band symmetry accordingly.
