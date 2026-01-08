# Newton interpolation and natural cubic spline 
# Compare interpolation of:
# (i) cos(x) on [0, pi] with 10 equally spaced points
# (ii) 1/(1+25 x^2) on [-1, 1] with 10 equally spaced points

import math
import matplotlib.pyplot as plt

def divided_differences(x, y):
    """
    recursively compute divided differences for Newton interpolation
    """
    n = len(x)
    coefs = y[:]  # will be modified in-place to store divided differences
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coefs[i] = (coefs[i] - coefs[i - 1]) / (x[i] - x[i - j])
    return coefs

def newton_eval(coefs, x_nodes, t):
    """
    evaluate Newton polynomial at t 
    """
    n = len(coefs)
    result = coefs[-1]
    for k in range(n - 2, -1, -1):
        result = result * (t - x_nodes[k]) + coefs[k]
    return result

def natural_cubic_spline(x, y):
    """
    自然三次样条, 解三对角矩阵 A c = alpha, 其中:
      - h_i = x_{i+1} - x_i
      - 未知 c_i = 1/2 * f''(x_i)(二阶导的一半)
      - A 的三对角元素(i=1..n-2):
            A[i,i-1] = h_{i-1}
            A[i,i]   = 2*(h_{i-1}+h_i)
            A[i,i+1] = h_i
      - 右端 alpha[i] = 3*((y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1})
        = (1/2) * 6*((y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1})
      - 自然边界: c_0 = c_n = 0

    Thomas算法变量含义:
      - l[i]: 前向消元后的主对角枢轴 = A[i,i] - A[i,i-1]*mu[i-1]
      - mu[i]: 上对角比例 = A[i,i+1]/l[i]
      - z[i]: 前向替换后的右端 = (alpha[i] - A[i,i-1]*z[i-1])/l[i]
    将A转为上三角矩阵后,回代:
      - c[j] = z[j] - mu[j]*c[j+1]
    之后区间系数均可得到 a + bx + cx^2 + dx^3: 
      - a[j] = y[j]
      - b[j] = (y[j+1]-y[j])/h[j] - h[j]*(2*c[j]+c[j+1])/3
      - d[j] = (c[j+1]-c[j])/(3*h[j])
    返回每个区间 [x_j,x_{j+1}] 的 (a,b,c,d) 及节点 x。
    """
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n - 1)]
    # alpha
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = 3.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    # l, mu, z  (for Thomas algorithm)
    l = [0.0] * n
    mu = [0.0] * n
    z = [0.0] * n
    l[0] = 1.0
    mu[0] = 0.0
    z[0] = 0.0
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
    c[n-1] = 0.0
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
        a[j] = y[j]
    # return spline coefficients per interval and the breakpoints
    return a, b, c[:-1], d, x  # c array truncated to interval count

def spline_eval(a, b, c, d, x_nodes, t):
    # find interval i such that x_nodes[i] <= t <= x_nodes[i+1]
    # a+bx+cx^2+dx^3
    n = len(x_nodes)
    if t <= x_nodes[0]:
        i = 0
    elif t >= x_nodes[-1]:
        i = n - 2
    else:
        # linear search (n is small)
        i = 0
        while i < n - 1 and not (x_nodes[i] <= t <= x_nodes[i+1]):
            i += 1
        if i == n - 1:
            i = n - 2
    dx = t - x_nodes[i]
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx

def run_case(f, a, b, n_points, title, fig_path=None):
    # nodes
    n = n_points
    xs = [a + i * (b - a) / (n - 1) for i in range(n)]
    ys = [f(x) for x in xs]
    # Newton polynomial
    coefs = divided_differences(xs, ys)
    # natural cubic spline
    a_s, b_s, c_s, d_s, x_nodes = natural_cubic_spline(xs, ys)
    # evaluation grid
    m = 400
    xs_dense = [a + i * (b - a) / (m - 1) for i in range(m)]
    y_true = [f(x) for x in xs_dense]
    y_newton = [newton_eval(coefs, xs, x) for x in xs_dense]
    y_spline = [spline_eval(a_s, b_s, c_s, d_s, xs, x) for x in xs_dense]
    # errors
    err_newton = max(abs(y_true[i] - y_newton[i]) for i in range(m))
    err_spline = max(abs(y_true[i] - y_spline[i]) for i in range(m))
    print(f"{title}: max abs error Newton = {err_newton:.6e}, cubic spline = {err_spline:.6e}")
    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(xs_dense, y_true, 'k-', label='True')
    plt.plot(xs_dense, y_newton, 'b--', label='Newton poly')
    plt.plot(xs_dense, y_spline, 'g-.', label='Cubic spline')
    plt.plot(xs, ys, 'ro', label='Nodes')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path,dpi=300) 

if __name__ == "__main__":
    # case (i) cos(x)
    run_case(lambda x: math.cos(x), 0.0, math.pi, 10, "cos(x) on [0, pi], 10 nodes", "../pic/case1_cos.png")
    # case (ii) 1/(1+25x^2)
    run_case(lambda x: 1.0 / (1.0 + 25.0 * x * x), -1.0, 1.0, 10, "1/(1+25x^2) on [-1,1], 10 nodes", "../pic/case2_runge.png")
