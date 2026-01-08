import math 

def f(x):
    return math.exp(-x**2)

def trapezoidal(f, a, b, n):
    """
    trapezoidal method integral
    f : integral function (expects scalar input)
    n: number of subintervals
    [a,b] : integral interval
    """
    h = (b - a) / n
    # generate sample points
    y0 = f(a)
    yn = f(b)
    # sum interior points
    s = 0.0
    for i in range(1, n):
        xi = a + i * h
        s += f(xi)
    return h * (0.5 * y0 + s + 0.5 * yn)

def romberg(f, a, b, max_level=4):
    """
    build Romberg table R as a list of lists
    R[i][k] corresponds to R_{i,k} in standard notation (0-based)
    i means 2^i subdivisions of interval; k means lowest error order O(h^(2k+2))
    """
    R = [[0.0 for _ in range(max_level)] for _ in range(max_level)]
    R[0][0] = trapezoidal(f, a, b, 1)
    for i in range(1, max_level):
        n = 2**i
        R[i][0] = trapezoidal(f, a, b, n)
        for k in range(1, i + 1):
            R[i][k] = R[i][k - 1] + (R[i][k - 1] - R[i - 1][k - 1]) / (4**k - 1)
    return R

def build_error_tables(R, true_value):
    """
    print:
    relative error 
    absolute error
    """
    n = len(R)
    abs_err = [[None for _ in range(n)] for _ in range(n)]
    rel_err = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(i + 1):
            val = R[i][k]
            ae = abs(val - true_value)
            re = ae / abs(true_value) if true_value != 0.0 else float('nan')
            abs_err[i][k] = ae
            rel_err[i][k] = re
    return abs_err, rel_err

def print_table(title, T, fmt=".10f"):
    print(title)
    n = len(T)
    for i in range(n):
        row = []
        for k in range(n):
            v = T[i][k]
            if v is None:
                row.append("")
            else:
                row.append(f"{v:{fmt}}")
        print("\t".join(row))

if __name__ == "__main__":
    max_level = 5
    R = romberg(f, 0.0, 1.0, max_level=max_level)

    true_value = 0.7468241328124271  # reference true value
    print_table("Romberg table (rows i, columns k):", R, fmt=".10f")

    abs_err, rel_err = build_error_tables(R, true_value)
    print_table("\nAbsolute error table:", abs_err, fmt=".3e")
    print_table("\nRelative error table:", rel_err, fmt=".3e")

    approx = R[max_level - 1][max_level - 1]
    print(f"\nFinal approximation R({max_level-1},{max_level-1}): {approx:.12f}")
    print(f"True value: {true_value:.12f}")
    print(f"Absolute error: {abs(approx - true_value):.3e}")
    print(f"Relative error: {abs(approx - true_value)/true_value:.3e}")