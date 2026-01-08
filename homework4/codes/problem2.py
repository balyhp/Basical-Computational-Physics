import math
import matplotlib.pyplot as plt
# 数据点
x = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
y = [14.6,18.5,36.6,30.8,59.2,60.1,62.2,79.4,99.9]

def gaussian_elimination(A, b):
    """
     A: list of n lists (n x n), b: list length n
     use Gaussian elimination to solve Ax = b ,
     here  Ax=b is a processed equ. from least-square partial over coeffeicients 
     of ax+b and ax^2+bx+c
    """
    n = len(b)
    A = [row[:] for row in A]
    b = b[:]
    for k in range(n):
        # 部分主元
        pivot = k
        maxv = abs(A[k][k])
        for i in range(k+1, n):
            if abs(A[i][k]) > maxv:
                maxv = abs(A[i][k]); pivot = i
        if maxv == 0:
            raise ValueError("Singular matrix")
        if pivot != k:
            A[k], A[pivot] = A[pivot], A[k]
            b[k], b[pivot] = b[pivot], b[k]
        # 消元
        for i in range(k+1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]
    # 回代
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = b[i]
        for j in range(i+1, n):
            s -= A[i][j]*x[j]
        x[i] = s / A[i][i]
    return x

def linear_least_squares(x, y):
    """
    matrix form of linear least-square fit for ax+b, A: 2*2
    """
    n = len(x)
    Sx = Sxx = Sy = Sxy = 0.0
    for xi, yi in zip(x,y):
        Sx += xi
        Sxx += xi*xi
        Sy += yi
        Sxy += xi*yi
    A = [[n, Sx],
         [Sx, Sxx]]
    b = [Sy, Sxy]
    a0, a1 = gaussian_elimination(A, b)
    return a0, a1

def quadratic_least_squares(x, y):
    """
    matrix form of quadratic least-square fit for ax^2+bx+c  A: 3*3
    """
    n = len(x)
    S = [0.0]*5  # S[k] = sum x^k for k=0..4
    Syx = [0.0]*3  # Syx[k] = sum y*x^k for k=0..2
    for xi, yi in zip(x,y):
        xp = 1.0
        for k in range(5):
            S[k] += xp
            xp *= xi
        xp = 1.0
        for k in range(3):
            Syx[k] += yi * xp
            xp *= xi
    A = [
        [S[0], S[1], S[2]],
        [S[1], S[2], S[3]],
        [S[2], S[3], S[4]]
    ]
    b = [Syx[0], Syx[1], Syx[2]]
    a0, a1, a2 = gaussian_elimination(A, b)
    return a0, a1, a2

def evaluate_poly(coeffs, xi):
    """
    given coeffs of polynomial, evaluate at xi
    """
    s = 0.0
    p = 1.0
    for c in coeffs:
        s += c * p
        p *= xi
    return s

def metrics(y, yhat):
    """
    compute SSE, RMSE, R^2
    """
    n = len(y)
    sse = sum((yi - yhi)**2 for yi,yhi in zip(y,yhat))  # sum of squared errors
    rmse = math.sqrt(sse / n)                           # root mean squared error
    ym = sum(y)/n
    sst = sum((yi - ym)**2 for yi in y)
    r2 = 1.0 - sse/sst if sst != 0 else float('nan')    # R-squared
    return sse, rmse, r2

def plot_results(x_data, y_data, linear_coeffs, quad_coeffs, fig_path):
    xmin, xmax = min(x_data), max(x_data)
    step = (xmax - xmin) / 200.0
    xp = []
    xv = xmin
    while xv <= xmax:
        xp.append(xv)
        xv += step
    linear_vals = [linear_coeffs[0] + linear_coeffs[1]*xi for xi in xp]
    quad_vals = [evaluate_poly(quad_coeffs, xi) for xi in xp]

    plt.figure(figsize=(7,5))
    plt.scatter(x_data, y_data, color="black", label="data points")
    plt.plot(xp, linear_vals, color="blue", label="linear")
    plt.plot(xp, quad_vals, color="red", label="quadratic")
    plt.xlabel("x (cm)")
    plt.ylabel("T (°C)")
    plt.title("least-square fit of temperature data")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)

if __name__ == "__main__":
    # linear
    a0, a1 = linear_least_squares(x,y)  # T = a0 + a1 * x
    yhat_lin = [a0 + a1*xi for xi in x]
    sse_l, rmse_l, r2_l = metrics(y, yhat_lin)

    # quadratic
    a0q, a1q, a2q = quadratic_least_squares(x,y)  # T = a0q + a1q*x + a2q*x^2
    yhat_quad = [a0q + a1q*xi + a2q*xi*xi for xi in x]
    sse_q, rmse_q, r2_q = metrics(y, yhat_quad)

    print(" T(x) = ax + b ")
    print(f" a = {a1:.6f}, b = {a0:.6f}")
    print(f" RMSE = {rmse_l:.6f}, R^2 = {r2_l:.6f}")
    print()
    print(" T(x) = ax^2 + b x + c ")
    print(f" a = {a2q:.6f}, b = {a1q:.6f}, c = {a0q:.6f}")
    print(f" RMSE = {rmse_q:.6f}, R^2 = {r2_q:.6f}")
    print()
    print(" x    |    y  |   ax+b   |   ax^2+bx+c")
    for xi, yi, yl, yq in zip(x, y, yhat_lin, yhat_quad):
        print(f"{xi:4.1f} {yi:8.3f} | {yl:8.3f} | {yq:8.3f}")
    plot_results(x, y, (a0, a1), (a0q, a1q, a2q), "../pic/problem2_fit.png")