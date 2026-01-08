import numpy as np
from scipy.integrate import simpson

"""
simpson is accrate for 3-order and below poly.

gaussian integration method is accrate:  n-point := (2n-1)-order poly. and below
"""

# =========================
# 1. 定义 3s 轨道径向波函数
# =========================
def R_3s(r, Z):
    n = 3
    rho = 2 * Z * r / n
    R = (1 / (9 * np.sqrt(3))) * (6 - 6 * rho + rho**2) * (Z ** 1.5) * np.exp(-rho / 2)
    return R

# =========================
# 2. 计算积分 ∫ |R(r)|^2 * r^2 dr
# =========================
def radial_integral(Z=14, r_max=40, N=2000, grid_type="uniform"):
    if grid_type == "uniform":
        # 等距网格
        r = np.linspace(0, r_max, N)
        dr = r[1] - r[0]
        f = np.abs(R_3s(r, Z)) ** 2 * r**2
        integral = simpson(f, dx=dr)

    elif grid_type == "nonuniform":
        # 非均匀网格：r = r0 * (exp(t) - 1)
        r0 = 0.0005
        t_max = np.log(r_max / r0 + 1)
        t = np.linspace(0, t_max, N)
        r = r0 * (np.exp(t) - 1)
        dr_dt = r0 * np.exp(t)
        f = np.abs(R_3s(r, Z)) ** 2 * r**2 * dr_dt
        integral = simpson(f, x=t)

    else:
        raise ValueError("grid_type must be 'uniform' or 'nonuniform'")

    return integral

# =========================
# 3. 主程序部分
# =========================
Z = 14
#r_max = 40
#
#for N in [200, 500, 1000, 2000, 4000]:
#    I_uniform = radial_integral(Z, r_max, N, grid_type="uniform")
#    I_nonuniform = radial_integral(Z, r_max, N, grid_type="nonuniform")
#    print(f"N = {N:5d} | Uniform: {I_uniform:.6f} | Nonuniform: {I_nonuniform:.6f}")

import matplotlib.pyplot as plt
r = np.linspace(0,5,300)
f = np.abs(R_3s(r, Z)) ** 2 * r**2
plt.figure()
plt.plot(r,f)
plt.xlabel('r')
plt.ylabel(r'$|R(r)|^2 r^2$')
plt.savefig('./p3.png')
