"""
Radial Schrödinger solver (atomic units):
  [-1/2 d^2/dr^2 + l(l+1)/(2 r^2) + V(r)] u(r) = E u(r),  with u(r)=r*R(r)

实现要点
- 均匀网格，二阶中心差分离散 -1/2 d^2/dr^2，得到对称三对角矩阵；
- 仅在本征分解时调用 scipy.linalg.eigh_tridiagonal 或回退到 numpy.linalg.eigh；
- 边界：u(r_min)=u(r_max)=0；归一化 ∫|u|^2 dr = 1。
"""

from __future__ import annotations
import os
import math
from typing import Callable, Tuple, List, Optional
import matplotlib.pyplot as plt

def trapz(y: List[float], x: List[float]) -> float:
    """梯形积分 ∫ y dx"""
    s = 0.0
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        s += 0.5 * (y[i] + y[i + 1]) * h
    return s


def build_radial_grid(r_min: float = 1e-4, r_max: float = 30.0, N: int = 1200) -> List[float]:
    """均匀径向网格"""
    if r_min <= 0.0:
        raise ValueError("r_min 必须为正以避免 r=0 奇点")
    if r_max <= r_min:
        raise ValueError("r_max 必须大于 r_min")
    if N < 50:
        raise ValueError("网格点数 N 建议不小于 50")
    h = (r_max - r_min) / (N - 1)
    return [r_min + i * h for i in range(N)]


# -------------------------- 势函数（纯 Python） --------------------------

def coulomb_potential(r: List[float], Z: float = 1.0) -> List[float]:
    """V(r) = -Z / r"""
    return [-(Z / ri) for ri in r]


def local_pseudopotential_Li(
    r: List[float],
    Z_ion: float = 3.0,
    r_loc: float = 0.4,
    C1: float = -14.0093922,
    C2: float = 9.5099073,
    C3: float = -1.7532723,
    C4: float = 0.0834586,
) -> List[float]:
    """
    V_loc(r) = -Z_ion/r * erf(r/(sqrt(2) r_loc)) +
               exp(-0.5 (r/r_loc)^2) * [C1 + C2 (r/r_loc)^2 + C3 (r/r_loc)^4 + C4 (r/r_loc)^6]
    """
    out: List[float] = []
    for ri in r:
        x = ri / r_loc
        erfterm = math.erf(ri / (math.sqrt(2.0) * r_loc))
        poly = C1 + C2 * x * x + C3 * (x ** 4) + C4 * (x ** 6)
        out.append(-Z_ion * erfterm / ri + math.exp(-0.5 * x * x) * poly)
    return out


# -------------------------- 中心差分三对角构造 --------------------------

def build_tridiag_cdiff(
    r: List[float],
    V_eff: List[float],
) -> Tuple[List[float], List[float], float]:
    """
    二阶中心差分离散 -1/2 d^2/dr^2：
      u''(r_i) ≈ (u_{i+1} - 2 u_i + u_{i-1}) / h^2
    则 -1/2 u'' → 对角: +1/h^2，次对角: -1/(2 h^2)
    H = -1/2 d^2/dr^2 + V_eff(r)
    返回：
      diag (长度 M=N-2), off (长度 M-1), h
    """
    N = len(r)
    if N < 3:
        raise ValueError("网格太少，至少 N >= 3")
    h = r[1] - r[0]
    M = N - 2  # 内点数

    diag = [0.0] * M
    off = [0.0] * (M - 1)
    inv_h2 = 1.0 / (h * h)
    off_val = -0.5 * inv_h2

    for i in range(M):
        diag[i] = inv_h2 + V_eff[i]
        if i < M - 1:
            off[i] = off_val

    return diag, off, h


# -------------------------- 三对角本征求解--------------------------

def eigh_tridiagonal_symmetric(
    diag: List[float],
    off: List[float],
    k: int,
) -> Tuple[List[float], List[List[float]]]:
    """
    优先用 SciPy 的三对角专用求解器；若不可用，回退到 numpy.linalg.eigh（仅此处使用）
    返回：k 个最小本征值与对应本征向量列（按列返回）
    """
    try:
        from scipy.linalg import eigh_tridiagonal  # type: ignore
        w, v = eigh_tridiagonal(diag, off, select='i', select_range=(0, k - 1))
        eigvals = [float(wi) for wi in w]
        M = len(diag)
        eigvecs: List[List[float]] = [[float(v[row, col]) for col in range(k)] for row in range(M)]
        return eigvals, eigvecs
    except Exception:
        import numpy as np  # type: ignore
        M = len(diag)
        H = np.zeros((M, M), dtype=float)
        for i in range(M):
            H[i, i] = diag[i]
            if i < M - 1:
                H[i, i + 1] = off[i]
                H[i + 1, i] = off[i]
        w, v = np.linalg.eigh(H)
        idx = np.argsort(w)[:k]
        w = w[idx]
        v = v[:, idx]
        eigvals = [float(wi) for wi in w]
        eigvecs = [[float(v[row, col]) for col in range(k)] for row in range(M)]
        return eigvals, eigvecs


# -------------------------- 主求解流程 --------------------------

def solve_radial(
    V_func: Callable[[List[float]], List[float]],
    l: int = 0,
    r_min: float = 1e-4,
    r_max: float = 30.0,
    N: int = 1200,
    n_states: int = 3,
) -> Tuple[List[float], List[float], List[List[float]], List[List[float]]]:
    """
    返回：
    - energies: 长度 n_states 的最低本征值（Hartree）
    - r: 径向网格（长度 N）
    - u_list: 每态的 u(r)（含边界，长度 N）
    - R_list: 每态的 R(r)=u/r（含边界，长度 N）
    """
    # 均匀网格
    r = build_radial_grid(r_min=r_min, r_max=r_max, N=N)
    rin = r[1:-1]
    M = len(rin)

    # 有效势：V_eff = V + l(l+1)/(2 r^2)（只在内点上取值）
    V_raw = V_func(rin)
    V_eff = [V_raw[i] + 0.5 * l * (l + 1) / (rin[i] * rin[i]) for i in range(M)]

    # 中心差分三对角矩阵
    diag, off, h = build_tridiag_cdiff(r, V_eff)

    # 解前 n_states 个本征对
    evals, evecs_interior = eigh_tridiagonal_symmetric(diag, off, k=n_states)

    # 组装含边界的 u，并归一化；再算 R=u/r
    u_list: List[List[float]] = []
    R_list: List[List[float]] = []
    for j in range(n_states):
        u_full = [0.0] * N
        for i in range(M):
            u_full[i + 1] = evecs_interior[i][j]  # 边界为 0

        # 归一化：∫ |u|^2 dr = 1
        norm2 = trapz([ui * ui for ui in u_full], r)
        if norm2 > 0.0:
            inv = 1.0 / math.sqrt(norm2)
            u_full = [ui * inv for ui in u_full]

        R_full = [0.0] * N
        for i in range(1,N):
            R_full[i] = u_full[i] / r[i] if r[i] != 0.0 else 0.0

        u_list.append(u_full)
        R_list.append(R_full)

    return evals, r, u_list, R_list

def plot_eigenstates(
    r: List[float],
    R_sets: List[List[List[float]]],
    E_sets: List[List[float]],
    titles: List[str],
    savepath: Optional[str] = None,
) -> None:
    ncols = len(R_sets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2), squeeze=False)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for j in range(ncols):
        ax = axes[0, j]
        Ej = E_sets[j]
        Rj = R_sets[j]
        for i, R in enumerate(Rj):
            # 从索引 1 开始绘制，避免与 r≈0 点的伪连线
            ax.plot(r[1:], R[1:], color=colors[i % len(colors)], lw=1.8,
                    label=f"{i+2}p: E={Ej[i]:.6f} Ha")
        ax.set_xlabel("r (bohr)")
        ax.set_ylabel("R(r) = u(r)/r")
        ax.set_title(titles[j])
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300)
    plt.show()

def main():
    # 参数
    l = 0
    r_min, r_max, N = 1e-4, 30.0, 1200 # N： 采样点数
    n_states = 3

    # 势函数
    V_H = lambda rr: coulomb_potential(rr, Z=1.0)
    V_Li = lambda rr: local_pseudopotential_Li(rr, Z_ion=3.0, r_loc=0.4)

    # 氢原子
    E_H, r, uH, RH = solve_radial(V_H, l=l, r_min=r_min, r_max=r_max, N=N, n_states=n_states)
    print("Hydrogen (Z=1), l=0, first 3 states (Hartree):")
    for i, e in enumerate(E_H, 1):
        print(f"  n~{i}: E = {e:.8f}  (exact s: -1/(2 n^2) = {-1/(2*i*i):.8f})")

    # Li 局域赝势
    E_Li, r2, uLi, RLi = solve_radial(V_Li, l=1, r_min=r_min, r_max=r_max, N=N, n_states=n_states)
    print("Li local pseudopotential, l=1, first 3 states (Hartree):")
    for i, e in enumerate(E_Li, 1):
        print(f"{i}s: E = {e:.8f}")

    # 绘图
    plot_eigenstates(
        r,
        R_sets=[RLi],
        E_sets=[E_Li],
        titles=["Li local pseudopotential"],
        savepath=os.path.join(os.path.dirname(__file__), "..", "pic", "test1.png"),
    )

if __name__ == "__main__":
    main()