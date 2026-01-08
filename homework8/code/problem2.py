"""
一维含时薛定谔方程：
    i ħ ∂Ψ/∂t = - (ħ^2 / 2m) ∂^2Ψ/∂x^2 + V(x) Ψ
方势阱：
    V(x) = 0,    a <= x <= b
           V0,   其他区域
初态（高斯波包，从左侧入射）：
    Ψ(x,0) = sqrt(1/pi) * exp(i k0 x - (x - xi0)^2 / 2)   （随后数值归一化）
"""
import cmath
import math
import time
import matplotlib.pyplot as plt

# --------------------------
# 工具与网格/物理参数
# --------------------------

def linspace(xmin, xmax, n):
    """生成均匀网格（列表），包含端点，共 n 个点。"""
    if n < 2:
        return [xmin]
    dx = (xmax - xmin) / (n - 1)
    return [xmin + i * dx for i in range(n)]

def norm_l2(psi, dx):
    """离散 L2 范数：∫ |psi|^2 dx ≈ sum |psi_i|^2 dx"""
    s = 0.0
    for v in psi:
        s += (v.real * v.real + v.imag * v.imag)
    return math.sqrt(s * dx)

def normalize(psi, dx):
    """数值归一化 Ψ，使 ∫ |Ψ|^2 dx = 1"""
    nrm = norm_l2(psi, dx)
    if nrm == 0.0:
        return psi
    scale = 1.0 / nrm
    return [v * scale for v in psi]

def probability_density(psi):
    """|Ψ|^2 列表"""
    return [(v.real * v.real + v.imag * v.imag) for v in psi]

def square_well_potential(xgrid, a, b, V0):
    """方势阱：区间 [a,b] 内为 0，外为 V0"""
    out = []
    for x in xgrid:
        if a <= x <= b:
            out.append(0.0)
        else:
            out.append(V0)
    return out

def gaussian_packet(xgrid, xi0, k0, sigma=1.0):
    """
    高斯波包：exp(i k0 x) * exp(-(x - xi0)^2 / (2 sigma^2))
    注：题目给的系数 sqrt(1/pi) 仅用于连续归一化，这里将再做数值归一化。
    """
    psi = []
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for x in xgrid:
        phase = 1j * k0 * x
        gauss = math.exp(- (x - xi0) * (x - xi0) * inv2s2)
        psi.append(cmath.exp(phase) * gauss)
    return psi

# --------------------------
# explicit scheme
# --------------------------

def explicit_step(psi, V, dx, dt, hbar, m):
    """
    单步显式时间推进：
    Ψ_i^{n+1} = Ψ_i^n + (i ħ Δt / (2 m Δx^2)) (Ψ_{i+1}^n - 2Ψ_i^n + Ψ_{i-1}^n) - (i Δt / ħ) V_i Ψ_i^n
    边界采用 Dirichlet: Ψ_0 = Ψ_{N-1} = 0
    """
    n = len(psi)
    nx_last = n - 1
    psi_new = [0j] * n
    coef_diff = 1j * hbar * dt / (2.0 * m * dx * dx)
    coef_pot = -1j * dt / hbar

    psi_new[0] = 0j
    psi_new[nx_last] = 0j

    for i in range(1, nx_last):
        lap = psi[i + 1] - 2.0 * psi[i] + psi[i - 1]
        psi_new[i] = psi[i] + coef_diff * lap + coef_pot * V[i] * psi[i]

    return psi_new

# --------------------------
# Crank–Nicolson scheme
# --------------------------

def thomas_solve(a, b, c, d):
    """
    解三对角方程：a_i y_{i-1} + b_i y_i + c_i y_{i+1} = d_i
    输入长度均为 M（内点个数），a[0] 与 c[-1] 可忽略（边界条件处理到 d 中或边界为 0）。
    返回 y（长度 M）。
    """
    M = len(b)
    # 拷贝，避免原地改动
    aa = a[:]
    bb = b[:]
    cc = c[:]
    dd = d[:]

    # 前消元
    for i in range(1, M):
        w = aa[i] / bb[i - 1]
        bb[i] = bb[i] - w * cc[i - 1]
        dd[i] = dd[i] - w * dd[i - 1]

    # 回代
    y = [0j] * M
    y[M - 1] = dd[M - 1] / bb[M - 1]
    for i in range(M - 2, -1, -1):
        y[i] = (dd[i] - cc[i] * y[i + 1]) / bb[i]
    return y

def crank_nicolson_step(psi, V, dx, dt, hbar, m):
    """
    单步 CN 时间推进：A ψ^{n+1} = B ψ^{n}
    记 r = i ħ Δt / (4 m Δx^2), s_i = i Δt V_i / (2 ħ)
    左端（A）三对角：
        a_i = -r,  b_i = 1 + 2r + s_i,  c_i = -r
    右端（Bψ^n）：
        d_i = r*Ψ_{i+1}^n + (1 - 2r - s_i)*Ψ_i^n + r*Ψ_{i-1}^n
    边界：Dirichlet Ψ_0=Ψ_{N-1}=0（已并入）
    """
    n = len(psi)
    nx_last = n - 1
    M = n - 2  # 内点个数
    if M <= 0:
        return psi[:]

    r = 1j * hbar * dt / (4.0 * m * dx * dx)

    # 预分配三对角与 RHS
    a = [0j] * M
    b = [0j] * M
    c = [0j] * M
    d = [0j] * M

    for j in range(M):
        i = j + 1  # 映射到全域索引
        s = 1j * dt * V[i] / (2.0 * hbar)
        a[j] = -r
        b[j] = 1.0 + 2.0 * r + s
        c[j] = -r

        psi_ip1 = psi[i + 1] if i + 1 <= nx_last else 0j
        psi_im1 = psi[i - 1] if i - 1 >= 0 else 0j
        d[j] = (r * psi_ip1) + (1.0 - 2.0 * r - s) * psi[i] + (r * psi_im1)

    # 边界值为 0，若非零需要将边界贡献加到 d[j] 上（此处为 0 略过）

    y = thomas_solve(a, b, c, d)

    psi_new = [0j] * n
    psi_new[0] = 0j
    psi_new[nx_last] = 0j
    for j in range(M):
        i = j + 1
        psi_new[i] = y[j]
    return psi_new

# --------------------------
# 主流程：测试方势阱 + 高斯波包
# --------------------------
def run_simulation(method="cn"):
    """
    运行一次仿真，返回快照与时空数据。
    method: 'cn' 或 'explicit'
    """
    # 物理常数（无量纲化：ħ=1, m=1）
    hbar = 1.0
    m = 1.0

    T_total = 4.5  # 总时长
    sample_times = [0.0, 1.0, 2.0, 3.0, 4.0]  # 统一的采样时间点（秒）

    # 空间网格
    x_min, x_max = -20.0, 20.0
    N = 801  # 网格点数（含端点）
    x = linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    # 方势阱参数
    a, b = -5.0, 5.0
    V0 = 10.0  # 阱外势高
    V = square_well_potential(x, a, b, V0)

    # 初态（高斯波包，从左向右）
    xi0 = -7.0
    k0 = 3.0
    sigma = 1.0
    psi0 = gaussian_packet(x, xi0, k0, sigma)
    psi0 = normalize(psi0, dx)

    # 时间步长
    # 显式稳定条件：ħ dt / (2 m dx^2) <= 1/2 => dt <= m dx^2 / ħ
    dt_exp_max = (m * dx * dx) / hbar
    if method == "explicit":
        dt = 0.3 * dt_exp_max   # 显式法安全系数
    else:
        dt = 0.8 * dt_exp_max   # CN 可更大，但仍保证数值稳定与精度

    # 由统一总时长计算步数，保证两方法模拟时长一致
    total_steps = int(round(T_total / dt))
    T_total = total_steps * dt  # 对齐到整数步的实际总时长

    # 采样设置：统一的绝对时间，映射为对应步数索引
    snapshot_steps = sorted(set(max(0, min(total_steps, int(round(t / dt)))) for t in sample_times))
    # 热图历史存储的步进（两方法一致地保存约 200 帧）
    history_stride = max(1, total_steps // 200)

    # 时间推进
    psi = psi0[:]
    densities_time = []   # 存 |Ψ|^2 的时间历史（用于热图）
    time_axis = []        # 对应时间
    snapshots = {}        # step -> (time, psi_copy)

    t0 = time.time()
    for nstep in range(total_steps + 1):
        t = nstep * dt

        # 记录
        if nstep in snapshot_steps:
            snapshots[nstep] = (t, psi[:])
        if nstep % history_stride == 0:
            densities_time.append(probability_density(psi))
            time_axis.append(t)

        # 推进一步（最后一步不再推进）
        if nstep == total_steps:
            break

        if method == "explicit":
            psi = explicit_step(psi, V, dx, dt, hbar, m)
            # 显式法适度重归一
            psi = normalize(psi, dx)
        else:
            psi = crank_nicolson_step(psi, V, dx, dt, hbar, m)

    elapsed = time.time() - t0
    print(f"[{method.upper()}] steps={total_steps}, dt={dt:.3e}, T_total={T_total:.3f}, dx={dx:.3e}, "
          f"elapsed={elapsed:.2f}s, final norm={norm_l2(psi, dx):.6f}")

    return {
        "x": x,
        "dx": dx,
        "dt": dt,
        "time_axis": time_axis,
        "densities_time": densities_time,
        "snapshots": snapshots,
        "V": V,
        "V0": V0,
        "a": a,
        "b": b
    }

def plot_snapshots(result, title_prefix, outfile):
    """
    绘制若干时刻的 |Ψ|^2(x,t) 及缩放后的势能曲线。
    """
    x = result["x"]
    V = result["V"]
    V0 = result["V0"]
    snapshots = result["snapshots"]

    plt.figure(figsize=(8, 5))
    # 计算势能缩放，使其大致与概率密度同量级
    # 使用最高一帧的最大密度作为尺度
    max_rho = 1e-12
    for _, psi_snap in snapshots.values():
        for v in psi_snap:
            r = v.real * v.real + v.imag * v.imag
            if r > max_rho:
                max_rho = r
    scale = 0.8 * max_rho / (V0 if V0 != 0 else 1.0)
    V_scaled = [vv * scale for vv in V]

    # 按时间排序绘制
    for step in sorted(snapshots.keys()):
        t, psi = snapshots[step]
        rho = probability_density(psi)
        plt.plot(x, rho, lw=1.2, label=f"t={t:.3f}")

    # 势能曲线
    plt.plot(x, V_scaled, 'k--', lw=1.0, label="V(x) scaled")
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x,t)|^2$")
    plt.title(f"{title_prefix} " + r"$|\psi|^2$ snapshots")
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"save as: {outfile}")

def plot_spacetime_density(result, title_prefix, outfile):
    """
    绘制时空热图：y 轴为时间，x 轴为空间，颜色为 |Ψ|^2。
    """
    x = result["x"]
    time_axis = result["time_axis"]
    densities_time = result["densities_time"]
    # densities_time: List[ List[|psi|^2 over x] ]，matplotlib 可直接 imshow 列表
    plt.figure(figsize=(8, 4.8))
    extent = [x[0], x[-1], time_axis[0], time_axis[-1]]
    im = plt.imshow(densities_time, origin="lower", aspect="auto", extent=extent, cmap="plasma")
    plt.colorbar(im, label="|Ψ|^2")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"{title_prefix}" + r"$|\psi|^2$ space-time")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"save as: {outfile}")

def main():
    # run CN
    res_cn = run_simulation(method="cn")
    plot_snapshots(res_cn, "Crank-Nicolson", "../pic/cn_snapshots.png")
    plot_spacetime_density(res_cn, "Crank-Nicolson", "../pic/cn_spacetime.png")
    # run explicit stable
    #res_exp = run_simulation(method="explicit")
    #plot_snapshots(res_exp, "Explicit stable", "../pic/explicit_snapshots.png")
    #plot_spacetime_density(res_exp, "Explicit stable", "../pic/explicit_spacetime.png")

if __name__ == "__main__":
    main()