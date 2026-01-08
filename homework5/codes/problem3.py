import math

def simpson(f , N , a, b):
    """
    simpson integral calculator
    """
    if N < 3 or N % 2 == 0:
        raise ValueError("Simpson: N must be odd >= 3")
    h = (b-a)/(N-1)
    x = [a+k*h for k in range(0,N)]
    i = 0
    integral = 0
    while i < (N-2):
        integral += (h/3)*(f(x[i])+4*f(x[i+1])+f(x[i+2]))
        i += 2
    return integral

def R_3s(r, Z):
    """
    define R3s(r)
    """
    n = 3.0
    rho = 2.0 * Z * r / n
    Z32 = Z * math.sqrt(Z)  # Z^{3/2}
    return (1.0 / (9.0 * math.sqrt(3.0))) * (6.0 - 6.0 * rho + rho * rho) * Z32 * math.exp(-rho / 2.0)


def radial_integral(Z=14, r_min=0, r_max=40, N=2000, grid_type="uniform"):
    """
    integral for 2 grids
    """
    if grid_type == "uniform":
        def f(r):
            val = R_3s(r, Z)
            return (val * val) * (r * r)
        integral = simpson(f, N=N, a=r_min,b=r_max)

    elif grid_type == "nonuniform":
        # r = r0 * (exp(t) - 1)
        r0 = 5e-4
        t_min = 0.0
        t_max = math.log(r_max / r0 + 1.0)

        def f_t(t):
            r = r0 * (math.exp(t) - 1.0)
            dr_dt = r0 * math.exp(t)
            val = R_3s(r, Z=Z)
            return (val * val) * (r * r) * dr_dt

        return simpson(f_t, N=N, a=t_min, b=t_max)
    else:
        raise ValueError("grid_type must be 'uniform' or 'nonuniform'")

    return integral

if __name__=="__main__":
    Z = 14
    r_max = 40
    r_min = 0
    
    for N in [201, 501, 1001, 2001, 4001]:
        I_uniform = radial_integral(Z, r_min, r_max, N, grid_type="uniform")
        I_nonuniform = radial_integral(Z, r_min, r_max, N, grid_type="nonuniform")
        print(f"N = {N:5d} | Uniform: {I_uniform:.6f} | Nonuniform: {I_nonuniform:.6f}")
