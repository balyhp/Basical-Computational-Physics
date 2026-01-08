######  problem1 #######

#part1  Bisection method 
def target(x):
    return x**3-5*x+3

def d_target(x):
    return 3*x**2 - 5

def bisec_solver(low, high, threshold=1e-4):
    """Bisection with tolerance-only stopping. Return (root_4dp, iters)."""
    fl, fh = target(low), target(high)
    if fl * fh >= 0:
        raise ValueError("bisec_solver requires f(low)*f(high) < 0")
    iters = 0
    while (high - low) > threshold:
        mid = (low + high) / 2.0
        fm = target(mid)
        # exact hit
        if fm == 0.0:
            low = high = mid
            iters += 1
            break
        if fl * fm < 0:
            high, fh = mid, fm
        else:
            low, fl = mid, fm
        iters += 1
    return round((low + high) / 2.0, 4), iters

#part2  Newton-Raphson method 
def newton_raphson(x0, tol=1e-14):
    """Newton-Raphson with tolerance-only stopping. Return (root_14dp, iters)."""
    x = float(x0)
    iters = 0
    while True:
        fx = target(x)
        if abs(fx) <= tol:
            return round(x, 14), iters
        dfx = d_target(x)
        if dfx == 0:
            # small nudge to escape flat derivative
            x = x + tol
            iters += 1
            continue
        x_new = x - fx / dfx
        iters += 1
        if abs(x_new - x) <= tol:
            return round(x_new, 14), iters
        x = x_new

#part3 hybrid method(N-R+Bisection)
def hybrid_root(a, b, tol=1e-14, dmin=1e-14): # this version is correct
    """
    Hybrid method that continues Newton steps when they are accepted,
    and only falls back to bisection when derivative is too small or
    a Newton trial goes outside the bracket.
    """
    fa, fb = target(a), target(b)
    if fa * fb >= 0:
        raise ValueError("hybrid_root requires f(a)*f(b) < 0")
    iters = 0
    x = 0.5 * (a + b)  # start at midpoint

    while True:
        fx = target(x)
        # convergence check
        if abs(fx) <= tol or 0.5 * (b - a) <= tol:
            return round(x, 14), iters

        dfx = d_target(x)
        # derivative too small -> do a bisection step
        if abs(dfx) <= dmin:
            mid = 0.5 * (a + b)
            fmid = target(mid)
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
            x = mid
            iters += 1
            continue

        # try one Newton step
        x_nr = x - fx / dfx
        # if Newton goes outside bracket, do bisection instead
        if not (a < x_nr < b):
            mid = 0.5 * (a + b)
            fmid = target(mid)
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
            x = mid
            iters += 1
            continue

        # accept Newton step and continue Newton from x_nr
        iters += 1
        f_nr = target(x_nr)
        # check convergence after accepted Newton step
        if abs(f_nr) <= tol or abs(x_nr - x) <= tol:
            return round(x_nr, 14), iters

        # update bracketing interval to keep the root enclosed
        if fa * f_nr < 0:
            b, fb = x_nr, f_nr
        else:
            a, fa = x_nr, f_nr

        # continue Newton from the new point
        x = x_nr

def hybrid_root_old(a, b, tol=1e-14, dmin=1e-14): # wrong !!!
    """
    Hybrid method:
    - x = (a+b)/2
    - if |f(x)|<=tol or (b-a)/2<=tol -> done
    - if |f'(x)|<=dmin -> Bisection shrink using x
    - else take one Newton step x_nr = x - f/ f'
        * if x_nr in (a,b): use x_nr to shrink [a,b] (and if |x_nr-x|<=tol -> done)
        * else: fall back to Bisection shrink using x
    """
    fa, fb = target(a), target(b)
    if fa * fb >= 0:
        raise ValueError("hybrid_root requires f(a)*f(b) < 0")
    iters = 0
    while True:
        x = (a + b) / 2.0
        fx = target(x)
        # convergence condition for bisec and NR
        if abs(fx) <= tol or 0.5 * (b - a) <= tol:
            return round(x, 14),iters

        dfx = d_target(x)
        if abs(dfx) <= dmin:
            # dfx = 0 , bisection method
            if fa * fx < 0:
                b, fb = x, fx
            else:
                a, fa = x, fx
            iters += 1
            continue
        
        x_nr = x - fx / dfx  # try one Newton step
        # if x_nr out of [a,b], return to bisection
        if not (a < x_nr < b):
            if fa * fx < 0:
                b, fb = x, fx
            else:
                a, fa = x, fx
            iters += 1
            continue

        # x_nr is inside (a,b): accept the Newton probe, count the iteration
        iters += 1

        # if NR is enough to converge end
        if abs(x_nr - x) <= tol:
            return round(x_nr, 14), iters

        f_nr = target(x_nr)
        if fa * f_nr < 0: # shrink [a,b] with NR, maintain the sign of 2 ends different
            b, fb = x_nr, f_nr
        else:
            a, fa = x_nr, f_nr
        # continue hybrid loop (next iteration will use new bracket/midpoint or further NR)

#=====================================================

if __name__ == "__main__":
    bound1= 0   # bracket of two positive roots, [0,1], [1,2] separately
    bound2= 1
    bound3= 2

    root2_bi, it_bi1 = bisec_solver(bound1, bound2)
    root3_bi, it_bi2 = bisec_solver(bound2, bound3)
    print(f"Bisection [0,1]: root={root2_bi:.4f}, iters={it_bi1}")
    print(f"Bisection [1,2]: root={root3_bi:.4f}, iters={it_bi2}")

    # Newton-Raphson: polish to 14 dp
    root2_nr, it_nr1 = newton_raphson(root2_bi)
    root3_nr, it_nr2 = newton_raphson(root3_bi)
    print(f"Newton Raphson root1: {root2_nr:.14f}, iters={it_nr1}")
    print(f"Newton Raphson root2: {root3_nr:.14f}, iters={it_nr2}")

    # Hybrid on the same intervals
    root2_h, it_h1 = hybrid_root(bound1, bound2, tol=1e-14)
    root3_h, it_h2 = hybrid_root(bound2, bound3, tol=1e-14)
    print(f"Hybrid [0,1]: {root2_h:.14f}, iters={it_h1}")
    print(f"Hybrid [1,2]: {root3_h:.14f}, iters={it_h2}")