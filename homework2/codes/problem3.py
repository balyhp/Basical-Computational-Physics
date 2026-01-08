import numpy as np

hbar = 1.054571817e-34      # J*s
m_e  = 9.1093837015e-31     # kg
eV   = 1.602176634e-19      # J
V0 = 10.0 * eV             # J
a   = 0.2 * 1e-9           # m
beta = np.sqrt(2*m_e*V0)/hbar     # 1/m
za   = beta * a                   # dimensionless

def E_from_k(k):
    return (hbar**2 * k**2) / (2*m_e)

def kappa_from_k(k):
    return np.sqrt(np.maximum(beta**2 - k**2, 0.0))

def f_even_u(u):
    # F_even(u) = u tan u - sqrt(za^2 - u^2)
    if not np.isfinite(u): 
        return np.nan
    return u*np.tan(u) - np.sqrt(max(za*za - u*u, 0.0))

def f_odd_u(u):
    # F_odd(u) = -u cot u - sqrt(za^2 - u^2)  (cot u = cos/sin)
    return -u/np.tan(u) - np.sqrt(max(za*za - u*u, 0.0))

def bisect_u(fun, lo, hi, tol=1e-12, max_it=200):
    """Bisection on a single monotone interval (no singularity inside)."""
    flo, fhi = fun(lo), fun(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return None
    if flo == 0.0: 
        return lo
    if fhi == 0.0: 
        return hi
    if flo * fhi > 0:
        # try a tiny inward nudge to restore opposite signs
        for s in (1e-6, 1e-5, 1e-4):
            lo2, hi2 = lo + s*(hi-lo), hi - s*(hi-lo)
            flo, fhi = fun(lo2), fun(hi2)
            if np.isfinite(flo) and np.isfinite(fhi) and flo*fhi < 0:
                lo, hi = lo2, hi2
                break
        else:
            return None
    for _ in range(max_it):
        mid = 0.5*(lo+hi)
        fm  = fun(mid)
        if abs(hi-lo) < tol or fm == 0.0:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5*(lo+hi)

def find_roots_segmented(tol=1e-12, eps=1e-6):
    """
    Use u = k a and exact segmentation:
      even on (nπ, nπ+π/2); odd on (nπ+π/2, (n+1)π),
    each interval clipped to u < za. Each valid segment yields at most one root.
    """
    ke_list, ko_list = [], []
    nmax = int(np.floor(za/np.pi)) + 2  # enough segments to cover u<za

    # even segments
    for n in range(nmax):
        lo = n*np.pi + eps
        hi = min(n*np.pi + 0.5*np.pi - eps, za - eps)
        if lo < hi:
            r = bisect_u(f_even_u, lo, hi, tol=tol)
            if r is not None:
                ke_list.append(r / a)

    # odd segments
    for n in range(nmax):
        lo = n*np.pi + 0.5*np.pi + eps
        hi = min((n+1)*np.pi - eps, za - eps)
        if lo < hi:
            r = bisect_u(f_odd_u, lo, hi, tol=tol)
            if r is not None:
                ko_list.append(r / a)

    return ke_list, ko_list

# normaliaztion of psi
def build_wavefunction(k, parity):
    kap = kappa_from_k(k)
    ka  = k*a
    if parity == 'even':
        # inside: psi = C cos(kx), outside: A e^{κx} (x<-a), D e^{-κx} (x>a)
        denom = (a + np.sin(2*ka)/(2*k) + (np.cos(ka)**2)/kap)   # 1/C^2
        C = 1.0/np.sqrt(denom)
        B = 0.0
        A = C*np.cos(ka)*np.exp(kap*a)
        D = C*np.cos(ka)*np.exp(kap*a)
        def psi(x):
            x = np.asarray(x)
            psi_in  = C*np.cos(k*x)
            psi_out = C*np.cos(ka)*np.exp(-kap*(np.abs(x)-a))
            return np.where(np.abs(x) <= a, psi_in, psi_out)
    else:  # odd
        # inside: psi = B sin(kx), outside: A e^{κx} (x<-a), D e^{-κx} (x>a)
        denom = (a - np.sin(2*ka)/(2*k) + (np.sin(ka)**2)/kap)   # 1/B^2
        B = 1.0/np.sqrt(denom)
        C = 0.0
        A = -B*np.sin(ka)*np.exp(kap*a)
        D = +B*np.sin(ka)*np.exp(kap*a)
        def psi(x):
            x = np.asarray(x)
            psi_in  = B*np.sin(k*x)
            psi_out = B*np.sign(x)*np.sin(ka)*np.exp(-kap*(np.abs(x)-a))
            return np.where(np.abs(x) <= a, psi_in, psi_out)
    # 返回归一化系数（此处 C 或 B 即为内区系数），以及完整 psi 和 ABCD
    N = 1.0  # overall normalization already absorbed into B/C
    return N, psi, A, B, C, D

def compute_bound_states(max_states=4, tol=1e-12):
    # bisection method to get k
    ke_list, ko_list = find_roots_segmented(tol=tol)

    states = []
    for k in ke_list:
        E = E_from_k(k)
        if E < V0:
            N, psi, A, B, C, D = build_wavefunction(k, 'even')
            states.append(dict(parity='even', k=k, E=E, E_eV=E/eV, psi=psi, N=N,
                               A=A, B=B, C=C, D=D))
    for k in ko_list:
        E = E_from_k(k)
        if E < V0:
            N, psi, A, B, C, D = build_wavefunction(k, 'odd')
            states.append(dict(parity='odd', k=k, E=E, E_eV=E/eV, psi=psi, N=N,
                               A=A, B=B, C=C, D=D))

    states.sort(key=lambda s: s['E'])
    return states[:max_states]

if __name__ == "__main__":
    states = compute_bound_states(max_states=4, tol=1e-12)
    if len(states) == 0:
        print("No eigen states found.")
    else:
        print("Bounded eigenstates (coefficients A,B,C,D and energy E):")
        for i, st in enumerate(states, 1):
            print(f"n={i:>2d}  parity={st['parity']:>4s}  E = {st['E_eV']:.12f} eV")
            print(f"    A = {st['A']:.6e},  B = {st['B']:.6e},  C = {st['C']:.6e},  D = {st['D']:.6e}")
        
        # --- overlay plot of the first three eigenstates on one figure ---
        import matplotlib.pyplot as plt
        xgrid = np.linspace(-5*a, 5*a, 2001)
        nplot = min(3, len(states))

        plt.figure(figsize=(9, 5))
        for i in range(nplot):
            st = states[i]
            psi = st['psi'](xgrid)
            plt.plot(
                xgrid*1e9, psi, lw=1.6,
                label=f"n={i+1}, {st['parity']}, E={st['E_eV']:.4f} eV"
            )

        # well boundaries
        plt.axvline(-a*1e9, color='k', ls='--', lw=1.0, alpha=0.7)
        plt.axvline( a*1e9, color='k', ls='--', lw=1.0, alpha=0.7)

        plt.xlabel("x (nm)")
        plt.ylabel(r"$\psi(x)$")
        plt.title("Finite square well: first three bound-state wavefunctions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./answer.png",dpi=300)