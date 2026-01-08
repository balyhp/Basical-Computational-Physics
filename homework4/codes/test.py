import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# === 你的自然三次样条函数 ===
def natural_cubic_spline(x, y):
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n - 1)]
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = 3.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    l = [0.0] * n
    mu = [0.0] * n
    z = [0.0] * n
    l[0] = 1.0
    z[0] = mu[0] = 0.0

    for i in range(1, n - 1):
        l[i] = 2.0 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n-1] = 1.0
    z[n-1] = 0.0
    c = [0.0] * n
    b = [0.0] * (n - 1)
    d = [0.0] * (n - 1)
    a = [0.0] * (n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
        a[j] = y[j]

    return a, b, c, d, x


# === 测试数据 ===
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])

# 计算你实现的样条
a, b, c, d, X = natural_cubic_spline(x, y)

# 构造一个评估函数（按区间插值）
def spline_eval(xq, a, b, c, d, X):
    """在给定xq上评估样条"""
    xq = np.atleast_1d(xq)
    yq = np.zeros_like(xq)
    for i, xi in enumerate(xq):
        # 找到所在区间
        if xi <= X[0]:
            j = 0
        elif xi >= X[-1]:
            j = len(X) - 2
        else:
            j = np.searchsorted(X, xi) - 1
        dx = xi - X[j]
        yq[i] = a[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3
    return yq

# 计算评估结果
x_dense = np.linspace(x[0], x[-1], 400)
y_custom = spline_eval(x_dense, a, b, c, d, X)

# 用 SciPy 自带样条验证
cs = CubicSpline(x, y, bc_type='natural')
y_scipy = cs(x_dense)

# === 绘图比较 ===
plt.figure(figsize=(8,5))
plt.plot(x, y, 'o', label='data points')
plt.plot(x_dense, y_custom, '--', label='your spline')
plt.plot(x_dense, y_scipy, '-', alpha=0.6, label='SciPy CubicSpline')
plt.legend()
plt.title('Natural Cubic Spline Comparison')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# === 数值误差检查 ===
diff = np.abs(y_custom - y_scipy)
print("最大绝对误差:", np.max(diff))
print("平均误差:", np.mean(diff))
