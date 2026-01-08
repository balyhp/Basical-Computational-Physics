#problem2 search g(x,y)=sin(x+y)+cos(x+2y) minimum 
from math import sin, cos, sqrt

# -------- small vector helpers (2D) --------
def dot(u, v):
    # inner product <u,v>
    return u[0]*v[0] + u[1]*v[1]

def add(u, v):
    # vector addition u+v
    return (u[0]+v[0], u[1]+v[1])

def sub(u, v):
    # vector subtraction u-v
    return (u[0]-v[0], u[1]-v[1])

def scale(a, u):
    # scalar-vector product a*u
    return (a*u[0], a*u[1])

def norm2(u):
    # Euclidean norm
    return sqrt(dot(u, u))

# -------- objective and gradient --------
def g_fn(xy):
    # g(x,y) = sin(x+y) + cos(x+2y)
    x, y = xy       # xy here is a shape=(0,2) list
    return sin(x + y) + cos(x + 2*y)

def g_grad(xy): 
    # gradient: [cos(x+y) - sin(x+2y), cos(x+y) - 2 sin(x+2y)]
    x, y = xy
    gx = cos(x + y) - sin(x + 2*y)
    gy = cos(x + y) - 2.0*sin(x + 2*y)
    return (gx, gy)

# ======================= Gradient Descent method ===========================
def steepest_descent(x0, lr=0.1, tol=1e-10, min_lr=1e-12):
    """
    Basic gradient descent to minimize g_fn(x,y).
    - Direction: d_k = -grad g(x_k)
    - Step:  if decrease, step update by step*0.5 
    - Stop when any holds:
        |grad|^2 <= tol,  |x_{k+1}-x_k|^2 <= tol,  |f_{k+1}-f_k| <= tol
    Returns: (x*, f*, iterations)
    """
    x = (float(x0[0]), float(x0[1]))
    f = g_fn(x)
    k = 0             #iteration times
    cur_lr = lr

    while True:
        g = g_grad(x)
        # gradient-based stopping
        if norm2(g) <= tol:
            return x, f, k

        # descent direction
        d = (-g[0], -g[1])

        # tentative step with simple backtracking on learning rate
        decreased = False
        while cur_lr >= min_lr:
            x_new = add(x, scale(cur_lr, d))
            f_new = g_fn(x_new)
            if f_new < f:  # accept only if objective decreases
                decreased = True
                break
            cur_lr *= 0.5  # reduce step and try again

        if not decreased:
            # step size too small and no descent: consider converged
            return x, f, k

        # progress checks
        dx = norm2(sub(x_new, x))
        df = abs(f_new - f)

        x, f = x_new, f_new
        k += 1

        # if progress is tiny, stop
        if dx <= tol or df <= tol:
            return x, f, k

        # gently enlarge lr for next round to avoid stopping reaching minimum
        cur_lr = min(cur_lr * 1.5, lr)


if __name__ == "__main__":
    xy0 = (7.4, -3.6)  # initial point, you can change it to other points to test
    learning_rate = 0.2  #  initial learning rate

    x_gd, f_gd, it_gd = steepest_descent(xy0, lr=learning_rate, tol=1e-12)
    print("Steepset Descent method,   initial point: ", xy0,"  lr: ",learning_rate)
    print(f"final (x,y) = ({x_gd[0]:.10f}, {x_gd[1]:.10f}),  min g(x,y) = {f_gd:.10f},  iters = {it_gd}")

    #xy01 = (13.4, -5.6)
    #learning_rate1 = 0.1  #  initial learning rate
#
    #x_gd, f_gd, it_gd = steepest_descent(xy01, lr=learning_rate1, tol=1e-12)
    #print("Steepset Descent method,   initial point: ", xy01,"  lr: ",learning_rate1)
    #print(f"final (x,y) = ({x_gd[0]:.10f}, {x_gd[1]:.10f}),  min g(x,y) = {f_gd:.10f},  iters = {it_gd}")