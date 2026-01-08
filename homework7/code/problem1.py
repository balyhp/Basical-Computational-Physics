"""
使用欧拉法、Midpoint（RK2 中点）法、RK4、Euler-Trapezoidal（Heun）法 4种方法数值求解单摆，
并绘制角度与总能量随时间的变化曲线
"""
import math
import matplotlib.pyplot as plt

def pendulum_rhs(t, y, g=9.81, L=1.0):
    """
    单摆微分方程
    y = [theta, omega]
    theta' = omega
    omega' = -(g/L) * sin(theta)
    """
    theta, omega = y
    return [omega, -(g / L) * math.sin(theta)]

def step_euler(t, y, h, rhs, *rhs_args):
    """显式欧拉法（一步）"""
    k1 = rhs(t, y, *rhs_args)
    return [y[0] + h * k1[0], y[1] + h * k1[1]]

def step_midpoint(t, y, h, rhs, *rhs_args):
    """中点法（RK2 Midpoint）"""
    k1 = rhs(t, y, *rhs_args)
    y_mid = [y[0] + 0.5 * h * k1[0], y[1] + 0.5 * h * k1[1]]
    k2 = rhs(t + 0.5 * h, y_mid, *rhs_args)
    return [y[0] + h * k2[0], y[1] + h * k2[1]]

def step_heun(t, y, h, rhs, *rhs_args):
    """
    Euler-Trapezoidal (predict and correct once), 
    so just Improved Euler
    """
    k1 = rhs(t, y, *rhs_args)
    y_euler = [y[0] + h * k1[0], y[1] + h * k1[1]]
    k2 = rhs(t + h, y_euler, *rhs_args)
    return [
        y[0] + 0.5 * h * (k1[0] + k2[0]),
        y[1] + 0.5 * h * (k1[1] + k2[1]),
    ]

def step_rk4(t, y, h, rhs, *rhs_args):
    """四阶 Runge-Kutta（RK4）"""
    k1 = rhs(t, y, *rhs_args)
    y2 = [y[0] + 0.5 * h * k1[0], y[1] + 0.5 * h * k1[1]]
    k2 = rhs(t + 0.5 * h, y2, *rhs_args)
    y3 = [y[0] + 0.5 * h * k2[0], y[1] + 0.5 * h * k2[1]]
    k3 = rhs(t + 0.5 * h, y3, *rhs_args)
    y4 = [y[0] + h * k3[0], y[1] + h * k3[1]]
    k4 = rhs(t + h, y4, *rhs_args)
    return [
        y[0] + (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        y[1] + (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
    ]

def integrate(y0, t0, t_end, step, rhs, stepper, *rhs_args):
    """
    通用积分器：返回时间列表、解轨迹列表
    """
    if step <= 0:
        raise ValueError("步长 step 必须为正数")
    n_steps = int(math.ceil((t_end - t0) / step))
    ts = [t0 + i * step for i in range(n_steps + 1)]
    ys = [list(y0)]
    t = t0
    y = list(y0)
    for _ in range(n_steps):
        y = stepper(t, y, step, rhs, *rhs_args)
        ys.append(y)
        t += step
    return ts, ys

def total_energy(theta, omega, m=1.0, g=9.81, L=1.0):
    """
    单摆总能量：E = m g L (1 - cos(theta)) + 0.5 m (L omega)^2
    """
    V = m * g * L * (1.0 - math.cos(theta))
    T = 0.5 * m * (L * omega) ** 2
    return T + V

def main(theta0 = 5.0, omega0 = 0.0,  t0 = 0.0, t_end = 20.0, step = 0.01, filepath=None):
    # 物理参数
    g = 9.81                # 重力加速度
    L = 1.0                 # 摆长
    m = 1.0                # 质量 
    theta0 = theta0        # 初始角度（弧度）
    omega0 = omega0        # 初始角速度
    y0 = [theta0, omega0]

    t0 = t0
    t_end = t_end
    step = step           # 步长

    methods = {
        "Euler": step_euler,
        "Midpoint (RK2)": step_midpoint,
        "Euler-Trapezoidal (Heun)": step_heun,
        "RK4": step_rk4,
    }

    styles = {
        "Euler": {"color": "#d62728", "linestyle": "--", "linewidth": 1.8, "alpha": 0.95, "markevery": 250},
        "Midpoint (RK2)": {"color": "#1f77b4", "linestyle": "-.", "linewidth": 1.8, "alpha": 0.95, "markevery": 260},
        "Euler-Trapezoidal (Heun)": {"color": "#2ca02c", "linestyle": ":", "linewidth": 2.0, "alpha": 0.95, "markevery": 270},
        "RK4": {"color": "#EFF46EE0", "linestyle": 'dashdot', "linewidth": 1.8, "alpha": 0.95, "markevery": 280},
    }

    results = {}

    # 积分
    for name, stepper in methods.items():
        ts, ys = integrate(y0, t0, t_end, step, pendulum_rhs, stepper, g, L)
        thetas = [state[0] for state in ys]
        omegas = [state[1] for state in ys]
        energies = [total_energy(th, om, m=m, g=g, L=L) for th, om in zip(thetas, omegas)]
        results[name] = {"t": ts, "theta": thetas, "omega": omegas, "E": energies}

    E0 = total_energy(theta0, omega0, m=m, g=g, L=L)
    plt.figure(figsize=(11, 8))
    # 子图1：角度-时间
    ax1 = plt.subplot(2, 1, 1)
    for name, res in results.items():
        ax1.plot(
            res["t"], res["theta"],
            label=name,
            **styles.get(name, {})
        )
    ax1.set_title(r"$\theta$ vs time")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("θ (rad)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2)
    # 子图2：能量-时间
    ax2 = plt.subplot(2, 1, 2)
    for name, res in results.items():
        ax2.plot(
            res["t"], res["E"],
            label=name,
            **styles.get(name, {})
        )
    ax2.axhline(E0, color="k", linestyle="--", linewidth=1.0, label="benchmark")
    ax2.set_title("energy vs time")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("E (J)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2)
    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, dpi=300)
    plt.show()

if __name__ == "__main__":
    # 可调参数
    main(theta0 = 10.0, omega0 = 0.0,  t0 = 0.0, t_end = 40.0, step = 0.01, filepath=None)
