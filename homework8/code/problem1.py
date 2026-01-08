"""
离散化的松弛法(Jacobi / Gauss-Seidel / SOR)求解二维矩形区域上的泊松方程: 
    ∇^2 φ(x, y) = -rho(x, y) / ε0
边界条件：在矩形边界 x=0, x=Lx, y=0, y=Ly 处给定 Dirichlet 边界值 φ。
(a) rho=0, φ(顶部)=1 V, 其他边 0, Lx=1 m, Ly=1.5 m
(b) rho/ε0=1 V/m^2, 四边界 φ=0, Lx=Ly=1 m
"""
import time
import matplotlib.pyplot as plt

def zeros_2d(nx, ny, value=0.0):
    """创建 (nx+1) x (ny+1) 的二维列表并填充初值 value。"""
    return [[value for _ in range(ny + 1)] for _ in range(nx + 1)]

def copy_2d(src):
    """深拷贝二维列表。"""
    return [row[:] for row in src]

def max_abs_diff(a, b):
    """计算两个同尺寸二维数组的最大绝对差。"""
    nx = len(a) - 1
    ny = len(a[0]) - 1
    m = 0.0
    for i in range(nx + 1):
        for j in range(ny + 1):
            d = abs(a[i][j] - b[i][j])
            if d > m:
                m = d
    return m

def plot_phi(phi_grid, Lx, Ly, title=None, cmap="plasma", savepath=None):
    """
    将求得的 φ 二维网格绘制为热力图。
    """
    nx = len(phi_grid) - 1
    ny = len(phi_grid[0]) - 1
    img = [[phi_grid[i][j] for i in range(nx + 1)] for j in range(ny + 1)]
    plt.figure(figsize=(6, 4.5))
    extent = [0.0, Lx, 0.0, Ly]
    im = plt.imshow(img, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    plt.colorbar(im, label="phi (V)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"φ 热力图已保存: {savepath}")
    else:
        plt.show()

def solve_poisson_relaxation(
    Lx,
    Ly,
    Nx,
    Ny,
    rho_func,
    epsilon0=1.0,
    bc_left=0.0,
    bc_right=0.0,
    bc_bottom=0.0,
    bc_top=0.0,
    method="sor",         # 'jacobi' | 'gauss_seidel' | 'sor'
    omega=1.5,            # SOR 过松弛因子，1 等价 GS
    tol=1e-6,
    max_iters=50000,
    report_interval=1000,
):
    """
    使用松弛法求解二维泊松方程。

    参数说明：
    - Lx, Ly: 几何尺寸
    - Nx, Ny: 网格分段数（含边界点总数为 Nx+1, Ny+1）
    - rho_func(x, y): 返回 rho(x,y)
    - epsilon0: 真空介电常数（可设为 1 以简化数值）
    - bc_left, bc_right, bc_bottom, bc_top: 四边 Dirichlet 边界 φ 值（常数）
    - method: 'jacobi', 'gauss_seidel', 'sor'
    - omega: SOR 松弛因子（method='sor' 时使用）
    - tol: 收敛阈值（基于两次迭代的最大点差）
    - max_iters: 最大迭代次数
    - report_interval: 打印进度间隔（迭代步数）
    返回：
    - phi: 二维列表，解 φ
    - iters: 实际迭代步数
    - last_diff: 最后一次迭代的最大变化
    """
    dx = Lx / Nx
    dy = Ly / Ny

    # 预计算系数，避免循环里重复除法
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    denom = 2.0 * (inv_dx2 + inv_dy2)  # 拉普拉斯离散的分母

    # 初始化 φ 与 ρ
    phi = zeros_2d(Nx, Ny, 0.0)
    rho = zeros_2d(Nx, Ny, 0.0)
    for i in range(Nx + 1):
        x = i * dx
        for j in range(Ny + 1):
            y = j * dy
            rho[i][j] = rho_func(x, y)

    # 设置边界条件（常数边界）
    for j in range(Ny + 1):
        phi[0][j] = bc_left
        phi[Nx][j] = bc_right
    for i in range(Nx + 1):
        phi[i][0] = bc_bottom
        phi[i][Ny] = bc_top

    # 迭代
    last_diff = float("inf")
    t0 = time.time()

    if method == "jacobi":
        # Jacobi 需要旧 φ 的全拷贝
        for it in range(1, max_iters + 1):
            old = copy_2d(phi)
            # 更新内点（跳过边界）
            for i in range(1, Nx):
                for j in range(1, Ny):
                    rhs = (old[i + 1][j] + old[i - 1][j]) * inv_dx2 \
                        + (old[i][j + 1] + old[i][j - 1]) * inv_dy2 \
                        + rho[i][j] / epsilon0
                    phi[i][j] = rhs / denom

            last_diff = max_abs_diff(phi, old)
            if it % report_interval == 0:
                print(f"[Jacobi] iter={it}, maxΔ={last_diff:.3e}")
            if last_diff < tol:
                print(f"[Jacobi] Converged at iter={it}, maxΔ={last_diff:.3e}, elapsed={time.time()-t0:.2f}s")
                return phi, it, last_diff

    elif method == "gauss_seidel":
        for it in range(1, max_iters + 1):
            max_change = 0.0
            # 就地更新，使用最新值（比 Jacobi 通常更快）
            for i in range(1, Nx):
                for j in range(1, Ny):
                    old_ij = phi[i][j]
                    rhs = (phi[i + 1][j] + phi[i - 1][j]) * inv_dx2 \
                        + (phi[i][j + 1] + phi[i][j - 1]) * inv_dy2 \
                        + rho[i][j] / epsilon0
                    new_ij = rhs / denom
                    phi[i][j] = new_ij
                    delta = abs(new_ij - old_ij)
                    if delta > max_change:
                        max_change = delta

            last_diff = max_change
            if it % report_interval == 0:
                print(f"[GS] iter={it}, maxΔ={last_diff:.3e}")
            if last_diff < tol:
                print(f"[GS] Converged at iter={it}, maxΔ={last_diff:.3e}, elapsed={time.time()-t0:.2f}s")
                return phi, it, last_diff

    elif method == "sor":
        # Successive Over-Relaxation: φ_new = φ_old + ω * (φ_GS - φ_old)
        # 其中 φ_GS 为不加权的 Gauss-Seidel 更新值
        if omega <= 0.0 or omega >= 2.0:
            print("[SOR] warning: omega should be in (0, 2)")
            omega = 1.8
        for it in range(1, max_iters + 1):
            max_change = 0.0
            for i in range(1, Nx):
                for j in range(1, Ny):
                    old_ij = phi[i][j]
                    rhs = (phi[i + 1][j] + phi[i - 1][j]) * inv_dx2 \
                        + (phi[i][j + 1] + phi[i][j - 1]) * inv_dy2 \
                        + rho[i][j] / epsilon0
                    gs_value = rhs / denom
                    new_ij = old_ij + omega * (gs_value - old_ij)
                    phi[i][j] = new_ij
                    delta = abs(new_ij - old_ij)
                    if delta > max_change:
                        max_change = delta

            last_diff = max_change
            if it % report_interval == 0:
                print(f"[SOR] iter={it}, maxΔ={last_diff:.3e}")
            if last_diff < tol:
                print(f"[SOR] Converged at iter={it}, maxΔ={last_diff:.3e}, elapsed={time.time()-t0:.2f}s")
                return phi, it, last_diff
    else:
        raise ValueError("unknown method use 'jacobi', 'gauss_seidel', or 'sor'")

    print(f"[{method}] 未在 max_iters={max_iters} 内收敛，最后 maxΔ={last_diff:.3e}, 总耗时={time.time()-t0:.2f}s")
    return phi, max_iters, last_diff

def run_case_a():
    """
    测试 (a):
    ρ(x,y)=0,
    φ(0,y)=φ(Lx,y)=φ(x,0)=0,
    φ(x,Ly)=1 V,
    Lx=1 m, Ly=1.5 m
    """
    Lx = 1.0
    Ly = 1.5
    Nx = 80     # 可调网格密度，越大越精细但越慢
    Ny = 120
    epsilon0 = 1.0  # ρ=0，本参数不影响结果
    def rho_zero(x, y):
        return 0.0

    phi, iters, diff = solve_poisson_relaxation(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        rho_func=rho_zero, epsilon0=epsilon0,
        bc_left=0.0, bc_right=0.0, bc_bottom=0.0, bc_top=1.0,
        method="sor", omega=1.9,  
        tol=1e-6, max_iters=100000, report_interval=1000
    )
    dx = Lx / Nx
    dy = Ly / Ny
    print(f"(a) complete={iters}, last step Δ={diff:.3e}")
    plot_phi(phi, Lx, Ly, title=r"Case A: $\varphi(x,y)$", savepath="case_a_phi.png")

def run_case_b():
    """
    测试 (b):
    ρ/ε0=1 V/m^2, 即 ρ(x,y) = ε0 * 1
    φ(0,y)=φ(Lx,y)=φ(x,0)=φ(x,Ly)=0,
    Lx=Ly=1 m
    """
    Lx = 1.0
    Ly = 1.0
    Nx = 100
    Ny = 100
    epsilon0 = 1.0  # 为了数值简化设为 1，此时 ρ=1
    def rho_const(x, y):
        # 因为设 ε0=1，且 ρ/ε0=1 => ρ=1
        return 1.0

    phi, iters, diff = solve_poisson_relaxation(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        rho_func=rho_const, epsilon0=epsilon0,
        bc_left=0.0, bc_right=0.0, bc_bottom=0.0, bc_top=0.0,
        method="sor", omega=1.9,
        tol=1e-6, max_iters=100000, report_interval=1000
    )
    dx = Lx / Nx
    dy = Ly / Ny
    print(f"(b) complete: iters={iters}, last step Δ={diff:.3e}")
    plot_phi(phi, Lx, Ly, title=r"Case B: $\varphi(x,y)$", savepath="case_b_phi.png")

if __name__ == "__main__":
    run_case_a()
    run_case_b()